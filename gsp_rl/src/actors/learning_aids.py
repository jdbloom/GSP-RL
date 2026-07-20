"""Network factory methods, learning algorithms, and hyperparameter management.

Contains the three-level class hierarchy base:
- Hyperparameters: Loads and stores all config values from YAML dict.
- NetworkAids(Hyperparameters): Factory methods (make_*_networks), learning
  algorithms (learn_*), action selection (*_choose_action), and memory
  management (sample_memory, store_transition).

Actor (in actor.py) inherits from NetworkAids, completing the chain:
Actor -> NetworkAids -> Hyperparameters.

See Also: docs/modules/actors.md, docs/algorithms.md
"""
from gsp_rl.src.networks import (
    DQN,
    DDQN,
    DDQN_SF,
    DDPGActorNetwork,
    DDPGCriticNetwork,
    RDDPGActorNetwork,
    RDDPGCriticNetwork,
    TD3ActorNetwork,
    TD3CriticNetwork,
    EnvironmentEncoder,
    AttentionEncoder,
    simnorm,
)
from gsp_rl.src.actors.feature_stats import (
    RunningStandardizer,
    actor_gsp_feature_weight_diag,
)

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam

import numpy as np
import logging

_learn_logger = logging.getLogger("stelaris.learn")


def _check_nan(value, name):
    """Raise RuntimeError if value is NaN or Inf. Works with floats and tensors.

    For tensors: ~T.isfinite() is a single GPU kernel (vs. separate isnan +
    isinf + OR = three kernels), and causes one device sync instead of the two
    that the old ``T.isnan(v).any() or T.isinf(v).any()`` pattern required.
    """
    if isinstance(value, T.Tensor):
        if (~T.isfinite(value)).any():
            raise RuntimeError(f"NaN/Inf detected in {name}: {value}")
    else:
        if not np.isfinite(value):
            raise RuntimeError(f"NaN/Inf detected in {name}: {value}")


Loss = nn.MSELoss()


def vicreg_variance_loss(h: T.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> T.Tensor:
    """VICReg variance term — hinge loss keeping per-dim std >= target_std.

    Part of Task 5 (Bardes, Ponce, LeCun ICLR 2022, arxiv 2105.04906).
    For a feature batch of shape (batch, D), computes std along batch dim for
    each of D features and penalizes those below target_std. Forces the
    representation to have a variance floor, attacking dimensional collapse.

    Args:
        h: Feature tensor of shape (batch, D).
        target_std: Minimum acceptable per-dim std. VICReg paper uses 1.0 but
            for scale-aware application in MSE regression we pass the running
            label std estimate to avoid saturating the tanh output.
        eps: Numerical floor for the std calculation.

    Returns:
        Scalar tensor loss. Zero when all dims already satisfy std >= target_std.
    """
    std = T.sqrt(h.var(dim=0) + eps)
    return T.mean(F.relu(target_std - std))


def vicreg_covariance_loss(h: T.Tensor) -> T.Tensor:
    """VICReg covariance term — off-diagonal penalty on the feature covariance.

    Part of Task 5 (Bardes, Ponce, LeCun ICLR 2022).
    Decorrelates feature dimensions by penalizing off-diagonal elements of the
    batch-wise covariance matrix. Normalized by D to make the coefficient
    approximately scale-independent across feature widths.

    Args:
        h: Feature tensor of shape (batch, D).

    Returns:
        Scalar tensor loss. Zero when features are perfectly decorrelated.
    """
    N, D = h.shape
    h_centered = h - h.mean(dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / max(N - 1, 1)
    # Zero out the diagonal, sum of squares on off-diagonal, divide by D
    off_diag = cov - T.diag(T.diagonal(cov))
    return (off_diag ** 2).sum() / D


def jepa_cosine_loss(pred: T.Tensor, target: T.Tensor, eps: float = 1e-8) -> T.Tensor:
    """Normalized / cosine latent loss for the JEPA predictor (flag GSP_JEPA_COSINE_LOSS).

    Replaces raw ``F.mse_loss`` between the predicted future latent and the
    (detached) EMA-target latent. SPR (Schwarzer et al. 2007.05929, Table/ablation)
    found the raw-MSE variant catastrophic (0.040) versus the normalized-cosine
    variant (0.415): projecting both latents onto the unit sphere makes the loss
    scale-invariant, so the objective can no longer be trivially minimized by
    shrinking the latent norm (a collapse mode).

    Computes, per sample, ``1 - cos(pred, target)`` (in ``[0, 2]``) and averages
    over the batch. Zero iff the projected prediction points in the same
    direction as the target; 2 iff exactly opposite.

    Args:
        pred:   Predicted future latent, shape (batch, D).
        target: Target latent (detached), shape (batch, D).
        eps:    Numerical floor for the L2 normalization.

    Returns:
        Scalar tensor loss in [0, 2].
    """
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    target_n = F.normalize(target, dim=-1, eps=eps)
    cos = (pred_n * target_n).sum(dim=-1)
    return (1.0 - cos).mean()


def gsp_l2er_loss(actor_net, states: T.Tensor, eps: float = 1e-8) -> T.Tensor:
    """Compute the differentiable effective-rank regularization loss for the GSP head.

    Runs a partial forward pass through the actor network's fc1 and fc2 layers,
    capturing the input activation tensor (pre-linear) at each layer. Then
    computes the soft effective rank of each layer's input covariance matrix as:

        erank(M) = exp( H(p) )    where H is Shannon entropy and
        p_i = s_i^2 / sum(s^2)   with s = svdvals(M)

    This is the Yang 2025 / He 2509.22335 soft-rank surrogate — fully
    differentiable via torch.linalg.svdvals, which backprops through the SVD.

    The returned loss is the *negative* sum of per-layer effective ranks:
        L2ER = -sum_l erank(input_cov_l)

    so that *subtracting* lambda * L2ER from the MSE loss is equivalent to
    maximizing effective rank (pushing the representations toward full rank).

    Only fc1 and fc2 inputs are regularized; the output projection (mu) is
    skipped — the output head is a scalar projector and its rank is inherently 1.

    Args:
        actor_net: A DDPGActorNetwork (must have .fc1, .fc2, .relu attributes).
        states: Input batch of shape (batch, input_size). Must be on the same
            device as actor_net.
        eps: Numerical floor for singular values to prevent log(0).

    Returns:
        Scalar tensor: -sum(erank_fc1, erank_fc2). Fully connected to the
        computation graph — backward() populates gradients on actor_net.parameters().
    """
    def _erank(M: T.Tensor) -> T.Tensor:
        """Soft effective rank of matrix M (batch, dim) via input covariance.

        torch.linalg.svdvals is not implemented on MPS (Apple Silicon) as of
        PyTorch 2.x. We move the matrix to CPU for the SVD, keeping the result
        on the original device so gradient flow is preserved through the .to()
        call. This is a small CPU detour (matrix is at most batch×dim, typically
        16×400) and does not break the autograd graph.
        """
        # Input covariance: (dim, dim) — centre the batch
        M_c = M - M.mean(dim=0, keepdim=True)
        # Use the batch matrix directly for SVD (batch, dim) rather than
        # forming (dim, dim) explicitly — avoids O(dim^2) memory and is
        # identical in spectral structure up to a 1/(N-1) scale that cancels
        # in the normalisation step.
        M_cpu = M_c.to('cpu')  # MPS fallback: svdvals requires CPU on Apple Silicon
        s = T.linalg.svdvals(M_cpu).to(M.device)
        s = T.clamp(s, min=eps)
        s2 = s ** 2
        p = s2 / s2.sum()
        # Shannon entropy (nats) then exp → effective rank
        H = -(p * p.log()).sum()
        return H.exp()

    # Partial forward: capture inputs at fc1 and fc2
    x_fc1 = states                                  # input to fc1
    pre_act1 = actor_net.fc1(x_fc1)
    if actor_net.use_layer_norm:
        pre_act1 = actor_net.ln1(pre_act1)
    x_fc2 = actor_net.relu(pre_act1)               # input to fc2

    erank_fc1 = _erank(x_fc1)
    erank_fc2 = _erank(x_fc2)

    # Return negative sum so *subtracting* lambda * L2ER maximises rank
    return -(erank_fc1 + erank_fc2)


class Hyperparameters:
    """Configuration container loaded from a YAML config dict.

    Maps YAML keys to instance attributes. Notable name mappings:
    - GSP_LEARNING_FREQUENCY -> self.gsp_learning_offset
    - REPLACE_TARGET_COUNTER -> self.replace_target_ctr

    Also initializes self.time_step = 0 (used by TD3 warmup).
    """
    def __init__(self, config):
        # Coerce numeric hyperparameters to float. They can arrive as STRINGS:
        # PyYAML 1.1 parses scientific notation without a decimal point (e.g.
        # `3e-05`, which json.dumps emits for 0.00003) as a str, which then
        # reaches optim.Adam(lr=...) and crashes ("'<=' not supported between
        # instances of 'float' and 'str'"). float() here makes every LR/discount
        # value robust regardless of how it was serialized upstream.
        self.gamma = float(config['GAMMA'])
        self.tau = float(config['TAU'])
        self.alpha = float(config['ALPHA'])
        self.beta = float(config['BETA'])
        self.lr = float(config['LR'])

        # Phase 4 — independent GSP head learning rate.
        # Default: same value as the trunk/actor LR (config['LR']), preserving
        # exact legacy behavior for all existing batches. When set to a different
        # value, the GSP prediction head's Adam optimizer uses gsp_head_lr while
        # the main action-network optimizer continues to use self.lr.
        # Only affects the GSP actor/head network; the GSP critic and target
        # networks are unchanged (they remain tied to self.lr).
        self.gsp_head_lr = float(config.get('GSP_HEAD_LR', self.lr))

        self.epsilon = config['EPSILON']
        self.eps_min = config['EPS_MIN']
        self.eps_dec = config['EPS_DEC']
        # EVAL_EPSILON: epsilon-greedy action noise injected ONLY at eval (test=True)
        # for DQN/DDQN. Default 0.0 => eval stays pure-greedy, bit-exact with prior
        # runs. >0 tests the noise-driven basin-escape hypothesis (whether uniform
        # action noise lets a wedged transport cluster unstick). None coerces to 0.0
        # (fail-loud: never `or`-default a valid 0.0).
        _eval_eps = config.get('EVAL_EPSILON', 0.0)
        self.eval_epsilon = float(0.0 if _eval_eps is None else _eval_eps)

        self.gsp_learning_offset = config['GSP_LEARNING_FREQUENCY'] #learn after every 1000 action network learning steps
        self.gsp_batch_size = config['GSP_BATCH_SIZE']

        self.batch_size = config['BATCH_SIZE']
        self.mem_size = config['MEM_SIZE']
        self.replace_target_ctr = config['REPLACE_TARGET_COUNTER']

        # --- Critic-divergence stabilizers (all default to legacy no-ops) -----
        # The gate task induces off-policy Q-value divergence: with MSE loss on
        # large-magnitude returns (per-robot returns in the thousands, gamma
        # ~0.99997), the DDQN critic loss escalates ~3 orders of magnitude over
        # training and the policy collapses after an early success peak. These
        # flags expose the standard divergence controls. Defaults reproduce the
        # exact prior behavior, so existing batches are unaffected.
        #   CRITIC_LOSS    : 'mse' (default) | 'huber' — Huber/SmoothL1 caps the
        #                    per-sample gradient, the strongest lever under Adam.
        #   GRAD_CLIP_NORM : >0 clips the Q-network grad-norm after backward.
        #   REWARD_SCALE   : multiplies the immediate reward in the TD target
        #                    (preserves the horizon, unlike lowering gamma).
        #   Q_TARGET_CLIP  : >0 clamps the bootstrap target to [-v, v], bounding
        #                    the overestimation runaway.
        self.critic_loss = str(config.get('CRITIC_LOSS', 'mse')).lower()
        self.grad_clip_norm = float(config.get('GRAD_CLIP_NORM', 0.0))
        self.reward_scale = float(config.get('REWARD_SCALE', 1.0))
        self.q_target_clip = float(config.get('Q_TARGET_CLIP', 0.0))
        # Recency-weighted replay: exponential half-life (in stores) for the
        # sampling probability.  0 (default) = OFF, uniform sampling,
        # bit-identical to all prior runs.  When > 0, recent transitions are
        # sampled exponentially more often — the primary stabilizer for
        # target-reset value disruption in gate training.
        self.recency_halflife = float(config.get('RECENCY_HALFLIFE', 0.0))
        # Polyak (soft) target-network update for DQN/DDQN q_next.
        # 0.0 (default) = OFF: preserve exact current behavior — hard copy
        # every REPLACE_TARGET_COUNTER learn steps (bit-identical to all prior
        # runs). When > 0, apply a soft Polyak update q_next ← tau*q_eval +
        # (1-tau)*q_next EVERY learn step and SKIP the periodic hard reset.
        # Mirrors the DDPG/TD3 update_DDPG_network_parameters() pattern using
        # param.data.copy_() in-place. DDPG/TD3 are unchanged.
        self.soft_target_tau = float(config.get('SOFT_TARGET_TAU', 0.0))
        # ReDo plasticity intervention (Sokar 2023): periodically recycle dormant
        # actor units (re-init incoming, zero outgoing, clear Adam). Default OFF —
        # inert. Targets the gate-training seed variance that dormancy drives
        # (dormancy<->success r=-0.54). REDO_FREQUENCY=0 also disables.
        self.redo_enabled = bool(config.get('REDO_ENABLED', False))
        self.redo_tau = float(config.get('REDO_TAU', 0.1))
        self.redo_frequency = int(config.get('REDO_FREQUENCY', 1000))
        # Single configurable critic-loss fn used by every value-bootstrapping
        # update (DQN/DDQN/DDPG/RDDPG/TD3). 'mse' reproduces the legacy global
        # MSELoss / F.mse_loss exactly. Applied via the helpers below so all five
        # algorithms get identical divergence control. (GSP prediction losses and
        # the actor policy-gradient loss are intentionally NOT touched — they are
        # not value-bootstrapping and do not exhibit the same divergence.)
        self._critic_loss_fn = (
            nn.SmoothL1Loss() if self.critic_loss == 'huber' else nn.MSELoss())

        # --- Successor-Features value head (GSP_SF_ENABLED) -------------------
        # Barreto 2017 (1606.05312): the DDQN Q-net predicts psi(s,a) in
        # R^(n_actions x d_phi) — the discounted future sum of a low-dim cumulant
        # phi — and Q(s,a) = psi(s,a) . w with w a learned reward-weight vector.
        # The prediction psi *is* the value, so zeroing psi zeroes Q (causal by
        # construction). Raw obs is KEPT as the net input. All default OFF — an
        # unset GSP_SF_ENABLED reproduces the exact DDQN path byte-for-byte.
        #   GSP_SF_ENABLED    : master switch (routes build + learn to the SF path).
        #   GSP_SF_PHI_DIM     : d_phi, the cumulant width. Default 1 (fallback phi
        #                       = [scalar reward]); the host passes a richer
        #                       d_phi-dim phi per step via store_agent_transition.
        #   GSP_SF_W_LR        : LR for the reward-weight w regression (default =
        #                       the actor LR).
        #   GSP_SF_W_TARGET    : 'reward' (default, one-step w.phi ~= r) or
        #                       'reward_to_go' (regress Q=psi.w to the discounted
        #                       bootstrap target — a value-scale-consistent readout).
        self.gsp_sf_enabled = bool(config.get('GSP_SF_ENABLED', False))
        self.gsp_sf_phi_dim = int(config.get('GSP_SF_PHI_DIM', 1))
        self.gsp_sf_w_lr = float(config.get('GSP_SF_W_LR', self.lr))
        self.gsp_sf_w_target = str(config.get('GSP_SF_W_TARGET', 'reward')).lower()

        self.gsp_e2e_enabled = bool(config.get('GSP_E2E_ENABLED', False))
        self.gsp_e2e_lambda = float(config.get('GSP_E2E_LAMBDA', 1.0))
        self.gsp_e2e_linear_output = bool(config.get('GSP_E2E_LINEAR_OUTPUT', False))
        # GSP_E2E_STOP_GRAD_FEATURE (default False): when True, DETACH the GSP
        # prediction where it is spliced into the actor's Q-input so the actor's
        # TD gradient can no longer flow back into the head. The head then trains
        # ONLY from its own supervised MSE loss (gsp_mse_loss stays on the raw,
        # un-detached prediction). This removes the head↔critic coupling that makes
        # the spliced feature non-stationary and prevents critic convergence. False
        # preserves the exact prior behavior (TD gradient flows into the head).
        self.gsp_e2e_stop_grad_feature = bool(
            config.get('GSP_E2E_STOP_GRAD_FEATURE', False)
        )

        # GSP_E2E_NORMALIZE_FEATURE (default False): standardize the spliced GSP
        # prediction to ~unit variance at the point it is concatenated into the
        # actor's Q-input. Motivation: the raw spliced scalar (std ~0.024) sits
        # among O(1) egocentric obs; ACTOR_USE_LAYER_NORM normalizes the whole
        # 32-vector (not per-feature), so the actor cannot learn to weight a
        # feature that small — the frozen_mean causal ablation was NULL at n=100.
        # When True, a single RunningStandardizer instance (self.gsp_feature_stats)
        # is shared between the acting splice (RL-CT agent.make_agent_state) and the
        # learn splices (learn_DDQN_e2e / learn_TD3_e2e): stats update ONLY during
        # learning, acting reads frozen stats. RL-CT's Agent subclasses this Actor,
        # so the same per-robot instance owns both paths — guaranteeing identical
        # standardization. Default False leaves gsp_feature_stats = None and every
        # splice byte-identical to today. See gsp_rl/src/actors/feature_stats.py.
        # NOTE: gsp_network_output is not known yet here (Actor.__init__ sets it
        # AFTER this settings super().__init__ returns), so the RunningStandardizer
        # instance is constructed in Actor.__init__ once the feature width (K) is
        # resolved. This block only parses the flag. Default False.
        self.gsp_e2e_normalize_feature = bool(
            config.get('GSP_E2E_NORMALIZE_FEATURE', False)
        )
        # GSP_E2E_NORMALIZE_EMA_HALFLIFE (default 0 = OFF -> legacy all-history
        # Welford, byte-identical to existing runs). When > 0, the
        # RunningStandardizer tracks a bias-corrected exponential moving
        # mean/variance whose per-batch weight halves every N UPDATE calls
        # (learn steps), so the stats follow the RECENT feature distribution.
        # Motivation (2026-07-09, live cells): at high GSP_E2E_LAMBDA (40000)
        # the head's early outputs are large/noisy and permanently inflate the
        # all-history running std — post-norm feature std reached only
        # 0.17-0.39 instead of ~1.0, silently re-shrinking the feature the
        # standardizer exists to normalize (defeats GSP_E2E_NORMALIZE_FEATURE).
        # Only meaningful when GSP_E2E_NORMALIZE_FEATURE is on. A YAML null
        # degrades to 0.0 via the `or`. See feature_stats.py for the numerics.
        self.gsp_e2e_normalize_ema_halflife = float(
            config.get('GSP_E2E_NORMALIZE_EMA_HALFLIFE', 0.0) or 0.0
        )
        # Placeholder; Actor.__init__ overwrites with a RunningStandardizer(dim=K)
        # when the flag is on. Stays None when off → every splice byte-identical.
        self.gsp_feature_stats = None

        # GSP_E2E_SPLICE_GAIN (default 1.0 = byte-identical no-op): a FIXED
        # constant multiplier applied to the spliced GSP feature as the LAST
        # transform before concatenation, at BOTH splice points (the learn
        # splices here and RL-CT's acting splice, which reads this same attr
        # through the shared Actor). The stateless salience lever replacing the
        # deprecated running standardizer: with source-scaled labels
        # (GSP_TRAJ_LABEL_SCALE) the head's output dispersion is capped by the
        # tanh/clamp bound at ~0.1-0.2 std, and the 2026-07-10 lstd probe
        # showed the actor ignores it (flips 1.5-2.6% ≈ the pre-normalize
        # baseline; coupling needed ~unit scale). A constant gain (e.g. 10 for
        # mature pred_std ~0.1) restores O(1) salience with nothing to
        # checkpoint, restore, or warm up — train/eval identical by
        # construction. Chosen from screen data, never adapted online. Applied
        # AFTER the (optional) standardizer so the two compose predictably.
        _gain_raw = config.get('GSP_E2E_SPLICE_GAIN', 1.0)
        # None (YAML null) degrades to the default; an explicit 0.0 is kept —
        # it is a legitimate severing endpoint of a gain sweep, not a falsy
        # accident to be remapped.
        self.gsp_e2e_splice_gain = float(
            1.0 if _gain_raw is None else _gain_raw
        )

        # GSP_E2E_UNIFIED_TARGET_ARITH (default False = byte-identical legacy,
        # golden-tested): route learn_DDQN_e2e's Bellman target through the
        # SAME _q_target arithmetic and critic-grad-clip treatment learn_DDQN
        # uses. The legacy E2E path computes `rewards + gamma * bootstrap`
        # directly — NO REWARD_SCALE, NO Q_TARGET_CLIP — and never calls
        # _clip_critic_grad on q_eval (only the GSP head clips, at a hardcoded
        # max_norm=1.0). Under the robust-training recipe (REWARD_SCALE 0.1,
        # Q_TARGET_CLIP 1000, GRAD_CLIP_NORM 10) this trains IC arms
        # (learn_DDQN) and GSP-N E2E arms under a 10x different effective
        # reward scale with the stabilizers engaged on one side only — any
        # penalty/reward arithmetic locked against _q_target (e.g. the
        # 2026-07-10 obstacle-contact pre-reg) is wrong for every E2E arm.
        # When True: target = _q_target(rewards, bootstrap) (reward_scale *
        # rewards + gamma * bootstrap, then the Q_TARGET_CLIP clamp) and
        # _clip_critic_grad(q_eval) after backward, exactly as learn_DDQN.
        # Scope notes (verified 2026-07-10): learn_DDQN_jepa_coupled and
        # learn_TD3_e2e ALREADY use _q_target/_clip_critic_grad (no bypass —
        # the flag deliberately does not touch them), and CRITIC_LOSS already
        # reaches the E2E loss via the DDQN net's own `loss` (actor.py threads
        # critic_loss into nn_args), so the flag closes the remaining
        # target-arithmetic + grad-clip gap only. The head-loss path (lambda *
        # gsp_mse_loss) is untouched either way. Default False preserves the
        # locked advantage-splice screen recipe and historical comparability.
        self.gsp_e2e_unified_target_arith = bool(
            config.get('GSP_E2E_UNIFIED_TARGET_ARITH', False)
        )
        # Single condition source for the CALLER's fail-loud startup line
        # (RL-CT Main.py, same pattern as gsp_splice_advantage_engaged /
        # batched_gsp_path_engaged): True iff an E2E run will actually route
        # its Bellman target through the unified arithmetic. Exposed as an
        # attribute because the `stelaris.learn` logger below has NO handler
        # in production — RL-CT's run logger is `stelaris.<exp_name>` with
        # propagate=False and logging.lastResort drops INFO — so the lines
        # below are silently discarded on every real run (they surface only
        # under pytest caplog). The authoritative per-run python.log line is
        # emitted by Main.py, keyed on this attribute.
        self.gsp_e2e_unified_arith_engaged = (
            self.gsp_e2e_enabled and self.gsp_e2e_unified_target_arith
        )
        # Fail-loud engaged-path assertion (one line per Actor construction,
        # only where the flag governs anything, i.e. E2E runs): state the
        # arithmetic the E2E TD target will actually use, with the stabilizer
        # values as consumed — grep it before trusting any IC-vs-E2E
        # comparison under a REWARD_SCALE recipe. (Test-visible only; see the
        # attribute comment above for why production runs rely on Main.py.)
        if self.gsp_e2e_enabled:
            if self.gsp_e2e_unified_arith_engaged:
                _learn_logger.info(
                    "GSP_E2E_UNIFIED_TARGET_ARITH: ENGAGED — E2E Q-target = "
                    "reward_scale(%s)*rewards + gamma(%s)*bootstrap, "
                    "Q_TARGET_CLIP=%s, critic grad clip=%s (parity with "
                    "learn_DDQN)",
                    self.reward_scale, self.gamma, self.q_target_clip,
                    self.grad_clip_norm,
                )
            else:
                _learn_logger.info(
                    "GSP_E2E_UNIFIED_TARGET_ARITH: off — legacy E2E Q-target "
                    "= rewards + gamma(%s)*bootstrap (no reward_scale, no "
                    "Q_TARGET_CLIP, no critic grad clip; plain-DDQN arms "
                    "apply reward_scale=%s q_target_clip=%s grad_clip=%s)",
                    self.gamma, self.reward_scale, self.q_target_clip,
                    self.grad_clip_norm,
                )

        # GSP_SPLICE_ADVANTAGE_ONLY (default False): dueling Q-head with the
        # spliced GSP prediction wired into the ADVANTAGE stream only.
        # Motivation (2026-07-09 Q-probe, stelaris
        # tools/analysis/q_sensitivity_probe.py): with the flat Q-head the actor
        # consumes the spliced prediction but its effect on Q is ~99.8%
        # common-mode (a state-value offset) and only ~0.2% differential — the
        # prediction is an action-invariant input, and value-offset is the
        # gradient-cheapest absorption, so argmax flips on only ~1-2% of states.
        # When True (DQN/DDQN + GSP splice only), the Q-net becomes dueling:
        # V(s) from the pred-EXCLUDED input, A(s,a) from the full input,
        # Q = V + A - mean(A) — common-mode absorption of the prediction is
        # architecturally forbidden; the feature can only express itself as
        # action preference. The E2E gradient path through the splice is
        # unchanged (the advantage trunk is the same fc1/fc2/fc3 the flat head
        # used; GSP_E2E_STOP_GRAD_FEATURE still governs detachment). Effective
        # gate + loud rejection of unsupported schemes live in
        # Actor.build_networks. Default False = byte-identical legacy flat head.
        self.gsp_splice_advantage_only = bool(
            config.get('GSP_SPLICE_ADVANTAGE_ONLY', False)
        )
        # GLOBAL_KNOWLEDGE (default False): mirrored into the config by the
        # host (RL-CT Main.py mirrors its --global_knowledge CLI flag, the same
        # #53-B single-condition-source pattern as INDEPENDENT_LEARNING). The
        # host layout under global knowledge is
        # [env_obs, pred(K), global_knowledge((R-1)*4)] with input_size
        # INCLUDING the global-knowledge width, so the advantage-only splice's
        # (input_size, K) pred-column span would point at the global-knowledge
        # TAIL instead of the prediction — Actor.build_networks reads this to
        # reject that combination loudly instead of letting V silently read
        # the prediction.
        self.global_knowledge = bool(config.get('GLOBAL_KNOWLEDGE', False))

        # H-13 closure: LayerNorm in the main DQN/DDQN action network's trunk.
        # Independent of GSP_USE_LAYER_NORM (which only affects the GSP head).
        # Default False preserves legacy behavior. See
        # docs/research/gsp-hypothesis-tracker.md H-13 for the rationale (j44 vs
        # j123: 96% vs 18% final success with same collapsed GSP head, suggesting
        # actor-side LN drives the difference).
        self.actor_use_layer_norm = bool(config.get('ACTOR_USE_LAYER_NORM', False))

        # Per-episode diagnostics instrumentation (FAU, weight norms, effective
        # rank, Q-gap, pred diversity). Default ON — learning-health visibility
        # should be the norm ("instrument from day one"), not an opt-in we forget
        # to set (we ran the entire 2026-06 stability investigation blind because
        # this defaulted off). The measurement is inert (no-grad forwards; backward
        # probes zero grads before/after) so it cannot perturb training. Set
        # ``DIAGNOSTICS_ENABLED: false`` to disable for a pure-throughput run.
        # Spec: docs/specs/2026-04-17-diagnostics-instrumentation.md.
        self.diagnostics_enabled = bool(config.get('DIAGNOSTICS_ENABLED', True))
        self.diagnostics_freeze_episode = int(config.get('DIAGNOSTICS_FREEZE_EPISODE', 50))
        self.diagnostics_cadence = int(config.get('DIAGNOSTICS_CADENCE', 10))
        self.diagnostics_batch_size = int(config.get('DIAGNOSTICS_BATCH_SIZE', 1024))
        # Optional critic-side diagnostics (weight norms). Default OFF — adds latency
        # and most plasticity signals of interest are on the actor/policy network.
        # Set to True for targeted investigations (e.g., checking whether critic
        # weight norms grow unboundedly during DDPG/TD3 training).
        self.diagnose_critic = bool(config.get('DIAGNOSE_CRITIC', False))

        # Gradient zero fraction (He 2603.21173 OCP Thm 1) — cheap; default ON.
        # Tracks the fraction of weight gradient entries near zero per named layer.
        # Should co-vary with FAU under OCP continuity conditions; divergence
        # indicates a measurement artifact or gradient flow shut-off without
        # activation collapse. Layers probed are taken from DIAGNOSTIC_PROFILE
        # 'grad_layers' key (defaults to 'fau_layers' if not specified).
        self.diagnose_grad_zero = bool(config.get('DIAGNOSE_GRAD_ZERO', True))

        # Activation churn (Tang 2506.00592 C-CHAIN) — cheap once state_dict
        # snapshots exist; default ON. If no before/after snapshots are provided
        # to compute_diagnostics(), this metric is silently skipped. Caller is
        # responsible for snapshotting via copy.deepcopy(net.state_dict()) around
        # a training step.
        self.diagnose_churn = bool(config.get('DIAGNOSE_CHURN', True))

        # KFAC Hessian effective rank (He 2509.22335 Thm 6.2) — most expensive of
        # the three new metrics; requires a full forward+backward pass plus covariance
        # matrix construction per layer. Default OFF; opt-in for targeted
        # investigations of Hessian rank collapse. Layers probed are taken from
        # DIAGNOSTIC_PROFILE 'kfac_layers' key (defaults to 'fau_layers' if not
        # specified).
        self.diagnose_kfac = bool(config.get('DIAGNOSE_KFAC', False))

        # H-14 GSP-minus ablation flag. When True, the GSP head still runs (gets
        # trained, produces predictions) but those predictions are REPLACED WITH
        # ZERO before concatenation into the actor's augmented observation. This
        # is the QMIP-minus pattern — same architecture, same training, signal
        # removed — the clean test of "does the GSP prediction contribute?".
        # Applied in RL-CollectiveTransport's agent.make_agent_state; the flag
        # only matters if the host code reads it (gsp-rl does not use the value
        # itself). Default False preserves legacy behavior.
        self.gsp_zero_out_signal = bool(config.get('GSP_ZERO_OUT_SIGNAL', False))

        # Candidate A — change what the GSP head predicts. Default 'delta_theta'
        # is the legacy collective-Δθ target used in all dissertation runs (and
        # the target that produced the head collapse documented in H-13/H-14).
        # 'future_prox' retargets each robot's head to predict its own proximity
        # K=GSP_PREDICTION_HORIZON steps ahead — non-self-referential because
        # prox is determined by environment geometry + multi-agent action, not
        # directly chosen by the robot's own action. The flag is read here; the
        # delayed-label buffer that produces (state_t, prox_{t+K}) training pairs
        # lives in the host code (RL-CollectiveTransport agent.py).
        self.gsp_prediction_target = str(config.get('GSP_PREDICTION_TARGET', 'delta_theta'))
        self.gsp_prediction_horizon = int(config.get('GSP_PREDICTION_HORIZON', 5))

        # GSP_OUTPUT_KIND — controls how many targets the GSP head predicts.
        # Motivated by He 2509.22335 Theorem 6.2: rank(Hessian) <= P - k_τ*(I+O+1).
        # Increasing output dim O directly raises the achievable Hessian rank,
        # potentially breaking the rank-1 collapse pattern observed with O=1.
        #
        # Supported values:
        #   'delta_theta_1d'        (default) O=1  — legacy Δθ scalar, backward compat
        #   'future_prox_1d'                  O=1  — per-agent future proximity scalar
        #   'cyl_kinematics_3d'               O=3  — (cyl_Δx, cyl_Δy, cyl_Δθ) per step
        #   'cyl_kinematics_goal_4d'          O=4  — above + group_centroid_Δ_to_goal
        #   'time_to_goal_1d'                 O=1  — remaining steps to success (or 0)
        #   'neighbor_force_1d'               O=1  — mean applied force-magnitude of
        #                                            the OTHER robots K steps ahead
        #                                            (delayed label; reuses the
        #                                            future_prox FIFO). Coordination-
        #                                            relevant target: correlates with
        #                                            per-robot reward-to-go at |0.33|
        #                                            vs delta_theta 0.06 (2026-07-04).
        #   'delta_theta_traj'                O=K  — payload-rotation TRAJECTORY over
        #                                            the next K steps: the size-K vector
        #                                            [Δθ(t→t+1), …, Δθ(t+K-1→t+K)], each
        #                                            per-step rotation wrap-safe to
        #                                            [-180,180) degrees (delayed label,
        #                                            reuses the future_prox FIFO). The
        #                                            head predicts the whole anticipated
        #                                            rotation PATH, not just the endpoint;
        #                                            neighbors ingest each other's size-K
        #                                            path and the actor Q-net head ingests
        #                                            it too. O is COUPLED to the horizon:
        #                                            O = GSP_PREDICTION_HORIZON. At K=1 it
        #                                            reduces to the legacy single-step
        #                                            rotation (2026-07-05).
        #   'goal_progress_traj'              O=K  — GLOBAL payload progress-to-goal
        #                                            TRAJECTORY over the next K steps:
        #                                            per-step delta of the payload's
        #                                            distance-to-goal (prev - curr,
        #                                            positive = toward goal — the exact
        #                                            quantity from cyl_kinematics_goal_4d's
        #                                            4th component), as a K-vector via
        #                                            the same delayed FIFO as
        #                                            delta_theta_traj. Labels are RAW
        #                                            meters — no magic scaling; lambda is
        #                                            set from measured label std (F15).
        #                                            O = GSP_PREDICTION_HORIZON (2026-07-09).
        #   'cyl_displacement_traj'           O=2K — GLOBAL payload displacement
        #                                            TRAJECTORY over the next K steps:
        #                                            per-step payload (Δx, Δy), flattened
        #                                            [Δx1,Δy1,…,ΔxK,ΔyK]. Same delayed
        #                                            FIFO, vector entries. RAW meters.
        #                                            O = 2 * GSP_PREDICTION_HORIZON
        #                                            (2026-07-09).
        #
        # gsp_output_size_effective is the O to use when building the GSP head.
        # The legacy gsp_output_size kwarg (from config['GSP_OUTPUT_SIZE']) is kept
        # for backward compat on non-GSP_OUTPUT_KIND runs; this field overrides it
        # when GSP_OUTPUT_KIND is set to a non-default value.
        #
        # A dict value of None marks a HORIZON-COUPLED kind whose output dim is not a
        # fixed constant but scales with GSP_PREDICTION_HORIZON (K) by the per-kind
        # multiplier in _GSP_TRAJ_KIND_HORIZON_MULTIPLIER. It is resolved from the
        # SAME config key that the RL-CollectiveTransport host (agent.py) uses to size
        # the actor/neighbor input, so head output width and input width always agree.
        _GSP_OUTPUT_KIND_SIZES = {
            'delta_theta_1d': 1,
            'future_prox_1d': 1,
            'cyl_kinematics_3d': 3,
            'cyl_kinematics_goal_4d': 4,
            'time_to_goal_1d': 1,
            'neighbor_force_1d': 1,
            # force_disagreement_1d: global directional-disagreement scalar of the
            # applied forces, 1 - |Σf_i|/Σ|f_i| (0=aligned, ~1=fighting). Same value
            # for every robot; non-redundant (a single robot can't infer the
            # collective's misalignment from its own view) — the top-ranked
            # in-setting target from the 2026-07-14 global-target sweep.
            'force_disagreement_1d': 1,
            'delta_theta_traj': None,       # size == K == GSP_PREDICTION_HORIZON
            'goal_progress_traj': None,     # size == K
            'cyl_displacement_traj': None,  # size == 2K (per-step Δx,Δy pairs)
        }
        # Horizon multiplier per horizon-coupled kind: output dim == mult * K.
        # cyl_displacement_traj packs (Δx, Δy) per step, hence 2 entries per step.
        _GSP_TRAJ_KIND_HORIZON_MULTIPLIER = {
            'delta_theta_traj': 1,
            'goal_progress_traj': 1,
            'cyl_displacement_traj': 2,
        }
        # The trajectory PREDICTION TARGETS (host-side delayed-FIFO labels) whose
        # names double as their required GSP_OUTPUT_KIND. Kept in lockstep with
        # RL-CollectiveTransport agent.py (_GSP_TRAJ_TARGETS) and Main.py.
        _GSP_TRAJ_TARGETS = (
            'delta_theta_traj', 'goal_progress_traj', 'cyl_displacement_traj'
        )
        self.gsp_output_kind = str(config.get('GSP_OUTPUT_KIND', 'delta_theta_1d'))
        if self.gsp_output_kind not in _GSP_OUTPUT_KIND_SIZES:
            raise ValueError(
                f"Unknown GSP_OUTPUT_KIND '{self.gsp_output_kind}'. "
                f"Valid values: {list(_GSP_OUTPUT_KIND_SIZES)}"
            )
        # Keep GSP_OUTPUT_KIND consistent with GSP_PREDICTION_TARGET. The host
        # (RL-CollectiveTransport agent.py) emits the label whose width is fixed
        # by the TARGET, while the head/buffer width here is fixed by the KIND.
        # For the trajectory targets the label is a size-(mult*K) vector but the
        # default kind 'delta_theta_1d' is scalar (O=1) — a mismatch that does NOT
        # fail at config time but crashes ~mid-episode deep in the replay buffer
        # with `could not broadcast (K,) into (1,)` (observed 2026-07-08, the
        # dtraj arm). The output dim is fully determined by the target, so
        # auto-derive the kind when it was left at the scalar default, and reject
        # an explicit contradiction loudly. (O=1 targets like future_prox are
        # dimensionally safe either way; this guards the horizon-coupled
        # trajectory targets.)
        if self.gsp_prediction_target in _GSP_TRAJ_TARGETS:
            if self.gsp_output_kind == 'delta_theta_1d':
                self.gsp_output_kind = self.gsp_prediction_target
            elif self.gsp_output_kind != self.gsp_prediction_target:
                raise ValueError(
                    f"GSP_PREDICTION_TARGET='{self.gsp_prediction_target}' requires "
                    f"GSP_OUTPUT_KIND='{self.gsp_prediction_target}' (the size-K "
                    "trajectory "
                    f"output); got GSP_OUTPUT_KIND='{self.gsp_output_kind}'."
                )
        _kind_size = _GSP_OUTPUT_KIND_SIZES[self.gsp_output_kind]
        if _kind_size is None:
            # Horizon-coupled kind: output dim == multiplier * GSP_PREDICTION_HORIZON.
            _horizon = int(self.gsp_prediction_horizon)
            if _horizon < 1:
                raise ValueError(
                    f"GSP_OUTPUT_KIND='{self.gsp_output_kind}' requires "
                    f"GSP_PREDICTION_HORIZON >= 1, got {_horizon}"
                )
            _kind_size = (
                _GSP_TRAJ_KIND_HORIZON_MULTIPLIER[self.gsp_output_kind] * _horizon
            )
        self.gsp_output_size_effective = _kind_size

        # Weight initialization scheme for the GSP head's hidden layers.
        # 'fanin' (default) preserves legacy behavior for all in-flight runs.
        # 'kaiming' uses Kaiming He normal init — see DDPGActorNetwork docstring.
        self.gsp_init_scheme = str(config.get('GSP_INIT_SCHEME', 'fanin'))

        # Phase 3 — effective-rank regularization on the GSP head.
        # When > 0 the MSE loss is reduced by lambda_er * sum(effective_rank per
        # layer), pushing activations toward higher-rank (less collapsed)
        # representations. See He et al. 2509.22335 for the theoretical grounding
        # (OCP Theorem 1) and the Stelaris Phase-3 experiment plan.
        # Default 0.0 → strict no-op, all historical runs unaffected.
        self.gsp_l2er_lambda = float(config.get('GSP_L2ER_LAMBDA', 0.0))

        # C-CHAIN churn-minimizing regularization on the GSP head
        # (Tang et al. 2506.00592, ICML 2025). When > 0, after the main MSE
        # (+ optional L2-ER) optimizer step, a second backward+step penalizes
        # the L2 change in head outputs on the same mini-batch:
        #   L_cchain = λ * F.mse_loss(head(states), pre_step_outputs.detach())
        # This counteracts plasticity loss by limiting how much each mini-batch
        # shifts the network's function on the training distribution.
        # Default 0.0 → strict no-op; all historical runs are unaffected.
        self.gsp_cchain_lambda = float(config.get('GSP_CCHAIN_LAMBDA', 0.0))

        self.noise = config['NOISE']
        self.update_actor_iter = config['UPDATE_ACTOR_ITER']
        self.warmup = config['WARMUP']
        self.time_step = 0
        # H-phase5-4: when True, skip the GSP head's optimizer step entirely.
        # Head stays at random init for the run. Default False preserves all
        # prior behavior. Read in actor.py:502 in the learn() loop.
        self.gsp_head_frozen = bool(config.get('GSP_HEAD_FROZEN', False))

        # JEPA (Joint Embedding Predictive Architecture) latent-space head.
        # When enabled, the legacy scalar future_prox prediction is replaced
        # by: online encoder → predictor → latent MSE against EMA target encoder.
        # The actor receives the 32-d encoder output instead of the 1-d gsp_pred.
        # Default False — all existing runs are unaffected.
        self.gsp_jepa_enabled = bool(config.get('GSP_JEPA_ENABLED', False))
        self.gsp_encoder_dim = int(config.get('GSP_ENCODER_DIM', 32))
        self.gsp_encoder_ema_tau = float(config.get('GSP_ENCODER_EMA_TAU', 0.995))

        # --- Coupled-JEPA levers (2026-07-05 literature-convergent fix) ---
        # All default OFF so an existing GSP_JEPA_ENABLED run is byte-identical.
        # See docs/research/2026-07-05-literature-synthesis-causal-prediction.md.
        #
        # (1) GSP_JEPA_COUPLE_VALUE — let the DDQN value-loss gradient flow into
        #     gsp_encoder_online (mirrors the learn_DDQN_e2e coupling). This is the
        #     core "make the latent decision-relevant" change (Ni et al. 2401.08898
        #     Thm 2: self-prediction ZP + value together induce reward-predictive
        #     representations; ZP alone stays decision-irrelevant). Requires the
        #     DDQN e2e replay path (gsp_obs stored) so the encoder can be re-run
        #     WITH gradient inside the actor learn step.
        self.gsp_jepa_couple_value = bool(config.get('GSP_JEPA_COUPLE_VALUE', False))
        #
        # (1b) GSP_JEPA_VALUE_STOPGRAD_ACTOR — the explicit resolution of the
        #     Ni-couple (#1) vs Dreamer-freeze (freeze world-model while updating
        #     the actor) tension. DDQN has NO separate actor: the Q-net is both the
        #     value function AND the policy (argmax_a Q). So "value-representation"
        #     and "actor-input" are literally the same spliced latent tensor.
        #     - False (default for the coupling arm): the value-loss gradient flows
        #       FULLY into gsp_encoder_online (pure Ni coupling).
        #     - True: the spliced latent is detached before entering the Q-net, so
        #       the encoder is shaped ONLY by its self-prediction loss (pure
        #       Dreamer-freeze) while the value head still consumes the latent.
        #     Exposed as a flag rather than silently picked — see the PR body.
        self.gsp_jepa_value_stopgrad_actor = bool(
            config.get('GSP_JEPA_VALUE_STOPGRAD_ACTOR', False)
        )
        # Weight on the DDQN value-loss contribution to the encoder when coupled.
        # Reuses the e2e lambda semantics; default 1.0 = plain sum.
        self.gsp_jepa_value_coef = float(config.get('GSP_JEPA_VALUE_COEF', 1.0))
        # Weight on the JEPA self-prediction (latent) loss inside the coupled step.
        # Default 1.0. Set to 0.0 to isolate the value path (used to verify the
        # Dreamer-freeze stop-grad blocks the value gradient to the encoder).
        self.gsp_jepa_selfpred_coef = float(config.get('GSP_JEPA_SELFPRED_COEF', 1.0))
        #
        # (2) GSP_JEPA_ACTION_COND — action-condition the predictor
        #     JEPAPredictor(z_t, a_t) → ẑ_{t+k} (SPR 2007.05929). Needs the action
        #     width; read from GSP_JEPA_ACTION_DIM (0 = legacy z-only predictor).
        self.gsp_jepa_action_cond = bool(config.get('GSP_JEPA_ACTION_COND', False))
        self.gsp_jepa_action_dim = int(config.get('GSP_JEPA_ACTION_DIM', 0))
        #
        # (3) GSP_JEPA_COSINE_LOSS — swap the raw latent MSE for the normalized /
        #     cosine latent loss (SPR: raw-MSE is the catastrophic variant).
        self.gsp_jepa_cosine_loss = bool(config.get('GSP_JEPA_COSINE_LOSS', False))

        # --- Latent-primary actor head (2026-07-06 pre-registration) ---
        # See docs/research/2026-07-06-latent-primary-actor-prereg.md (Stelaris).
        # Both default OFF so an existing GSP_JEPA_ENABLED / coupled run is
        # byte-identical (network dims, replay dims, config hash unchanged).
        #
        # (A) GSP_ACTOR_LATENT_PRIMARY — feed the actor's Q-net
        #     [latent(enc_dim) | neighbor/global latents] and DROP the raw env-obs
        #     block (Dreamer / TD-MPC2 style). The value gradient (already coupled
        #     via GSP_JEPA_COUPLE_VALUE) then MUST route through the encoder,
        #     forcing the policy to use the latent's moment-to-moment content
        #     rather than solving from raw obs and treating the latent as an
        #     optional side-feature (the frozen_mean-inert failure of the concat
        #     design). Changes network_input_size (actor.py) — the runtime
        #     augmented-obs builder (RL-CollectiveTransport make_agent_state) and
        #     the coupled splice (learn_DDQN_jepa_coupled) drop env_obs in lockstep.
        self.gsp_actor_latent_primary = bool(
            config.get('GSP_ACTOR_LATENT_PRIMARY', False)
        )
        #
        # (B) GSP_JEPA_SIMNORM — project the latent onto a per-group simplex
        #     (DreamerV3 2301.04104 / TD-MPC2 2310.16828) before it enters the
        #     Q-net AND before the self-prediction comparison. Applied inside
        #     JEPAEncoder.forward so the online latent, the EMA-target latent, and
        #     the host's choose_agent_gsp latent are all bounded/consistent via one
        #     code path. Bounds the heterogeneous latent input and stabilizes the
        #     coupled latent→value path.
        self.gsp_jepa_simnorm = bool(config.get('GSP_JEPA_SIMNORM', False))
        self.gsp_jepa_simnorm_group_size = int(
            config.get('GSP_JEPA_SIMNORM_GROUP_SIZE', 8)
        )

class NetworkAids(Hyperparameters):
    """Network factory, learning algorithms, action selection, and memory management.

    All methods operate on a 'networks' dict (plain dict, not a class) that
    contains the neural networks, replay buffer, learning scheme string, and
    step counter. This dict is either self.networks or self.gsp_networks,
    passed explicitly to allow the same learn/action methods to serve both
    the main action network and the GSP prediction network.
    """
    def __init__(self, config):
        super().__init__(config)

    # --- shared critic-divergence stabilizer helpers ----------------------
    # Applied uniformly across every value-bootstrapping update. With default
    # flags (reward_scale=1, q_target_clip=0, grad_clip_norm=0, critic_loss=mse)
    # each helper is a no-op / identity, so all five algorithms reproduce their
    # exact prior behavior.
    def _q_target(self, rewards, bootstrap):
        """Reward-scaled, optionally target-clipped Bellman target.
        `rewards` and `bootstrap` must already be broadcast-compatible (callers
        match the original per-algorithm shaping)."""
        if self.reward_scale == 1.0:
            target = rewards + self.gamma * bootstrap
        else:
            target = self.reward_scale * rewards + self.gamma * bootstrap
        if self.q_target_clip > 0:
            target = T.clamp(target, -self.q_target_clip, self.q_target_clip)
        return target

    def _clip_critic_grad(self, *nets):
        """Clip critic grad-norm in place after backward() when enabled."""
        if self.grad_clip_norm > 0:
            for net in nets:
                T.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip_norm)

    def _maybe_redo(self, net, probe_states, step) -> None:
        """Run ReDo dormant-unit recycling on the actor trunk every
        REDO_FREQUENCY learn steps when enabled. Uses the current minibatch
        states as the dormancy probe. Inert unless REDO_ENABLED and the net has
        the fc1/fc2/fc3 trunk (DQN/DDQN)."""
        if not self.redo_enabled or self.redo_frequency <= 0:
            return
        if step % self.redo_frequency != 0:
            return
        if not (hasattr(net, "fc1") and hasattr(net, "fc2") and hasattr(net, "fc3")):
            return
        from gsp_rl.src.actors.plasticity import redo_reset
        redo_reset(net, probe_states, [("fc1", "fc2"), ("fc2", "fc3")],
                   tau=self.redo_tau)

    def make_DQN_networks(self, nn_args):
        return {'q_eval':DQN(**nn_args), 'q_next':DQN(**nn_args)}
    
    def make_DDQN_networks(self, nn_args):
        return {'q_eval':DDQN(**nn_args), 'q_next':DDQN(**nn_args)}

    def make_DDQN_SF_networks(self, nn_args):
        """Build the Successor-Features DDQN pair (GSP_SF_ENABLED).

        Identical trunk to make_DDQN_networks; each net is a DDQN_SF that outputs
        psi (n_actions x d_phi) and carries its own learned reward-weight w. nn_args
        must include 'd_phi'. The reward weight w lives on q_eval and is trained by
        reward regression; q_next.w is Polyak/hard-copied with the rest of q_next.
        """
        return {'q_eval': DDQN_SF(**nn_args), 'q_next': DDQN_SF(**nn_args)}

    def make_DDPG_networks(self, actor_nn_args, critic_nn_args):
        DDPG_networks = {
                        'actor': DDPGActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': DDPGActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic': DDPGCriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic': DDPGCriticNetwork(**critic_nn_args, name = 'target_critic_1')}
        return DDPG_networks

    def make_TD3_networks(self, actor_nn_args, critic_nn_args):
        TD3_networks = {
                        'actor': TD3ActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': TD3ActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic_1': TD3CriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic_1': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_1'),
                        'critic_2': TD3CriticNetwork(**critic_nn_args, name = 'critic_2'),
                        'target_critic_2': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_2')}
        return TD3_networks
    
    def make_RDDPG_networks(self, lstm_nn_args, actor_nn_args, critic_nn_args):
        shared_ee = EnvironmentEncoder(**lstm_nn_args)
        RDDPG_networks = {
            'actor': RDDPGActorNetwork(shared_ee, DDPGActorNetwork(**actor_nn_args, name='actor')),
            'target_actor': RDDPGActorNetwork(EnvironmentEncoder(**lstm_nn_args), DDPGActorNetwork(**actor_nn_args, name='target_actor')),
            'critic': RDDPGCriticNetwork(shared_ee, DDPGCriticNetwork(**critic_nn_args, name = 'critic')),
            'target_critic':RDDPGCriticNetwork(EnvironmentEncoder(**lstm_nn_args), DDPGCriticNetwork(**critic_nn_args, name = 'target_critic'))
        }
        return RDDPG_networks
    
    def make_Environmental_Encoder(self, nn_args):
        lstm_networks = {'ee': EnvironmentEncoder(**nn_args)}
        return lstm_networks

    def make_Attention_Encoder(self, nn_args):
        Attention_networks = {'attention': AttentionEncoder(**nn_args)}
        return Attention_networks

    def update_DDPG_network_parameters(self, tau, networks):
        # Update Actor Network
        for target_param, param in zip(networks['target_actor'].parameters(), networks['actor'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # Update Critic Network
        for target_param, param in zip(networks['target_critic'].parameters(), networks['critic'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        return networks

    def update_TD3_network_parameters(self, tau, networks):
        actor_params = networks['actor'].named_parameters()
        critic_1_params = networks['critic_1'].named_parameters()
        critic_2_params = networks['critic_2'].named_parameters()
        target_actor_params = networks['target_actor'].named_parameters()
        target_critic_1_params = networks['target_critic_1'].named_parameters()
        target_critic_2_params = networks['target_critic_2'].named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        networks['target_critic_1'].load_state_dict(critic_1)
        networks['target_critic_2'].load_state_dict(critic_2)
        networks['target_actor'].load_state_dict(actor)

        return networks

    def DQN_DDQN_choose_action(self, observation, networks):
        state = T.tensor(observation, dtype = T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(state)
        return T.argmax(action_values).item()

    def DQN_DDQN_choose_action_batch(self, observations, networks):
        """Batched action selection for DQN/DDQN. Returns list of action indices."""
        states = T.tensor(np.array(observations), dtype=T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(states)
        return T.argmax(action_values, dim=1).cpu().tolist()
    
    def DDPG_choose_action(self, observation, networks):
        if networks['learning_scheme'] == 'RDDPG':
            # if using LSTM we need to add an extra dimension
            state = T.tensor(np.array(observation), dtype=T.float).to(networks['actor'].device)
            mu, _ = networks['actor'].forward(state)
            return mu.unsqueeze(0)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            return networks['actor'].forward(state).unsqueeze(0)
        
    
    def DDPG_choose_action_batch(self, observations, networks):
        """Batched action selection for DDPG. Returns (batch, output_size) numpy array."""
        if networks['learning_scheme'] == 'RDDPG':
            # RDDPG uses sequences — cannot batch across robots (stateful LSTM)
            raise NotImplementedError("RDDPG cannot be batched — use sequential choose_action")
        states = T.tensor(np.array(observations), dtype=T.float).to(networks['actor'].device)
        return networks['actor'].forward(states)

    def TD3_choose_action_batch(self, observations, networks, n_actions):
        """Batched action selection for TD3. Returns list of (1, output_size) numpy arrays.

        SEMANTIC DIVERGENCE from sequential TD3_choose_action — not just fp
        drift, so the #53-B float-drift-only equivalence claim
        (choose_actions_batch docstring) does NOT cover TD3:
          - self.time_step advances once per BATCH here vs once per ROBOT in
            the sequential loop → under batching the warmup phase ends R×
            sooner in env steps;
          - warmup actions are one (R, n_actions) np.random.normal draw vs R
            separate (n_actions,) draws, and exploration noise is one
            mus-shaped draw vs R sequential scalar draws — different shapes
            and stream order, so trajectories diverge whenever noise > 0 or
            warmup > 0.
        No host routes TD3 through the batched path today. Any future TD3 use
        requires its own equivalence work (warmup accounting, RNG-contract
        tests, re-baseline) before activation.
        """
        if self.time_step < self.warmup:
            batch_size = len(observations)
            mus = T.tensor(np.random.normal(scale=self.noise, size=(batch_size, n_actions)),
                           dtype=T.float).to(networks['actor'].device)
        else:
            states = T.tensor(np.array(observations), dtype=T.float).to(networks['actor'].device)
            mus = networks['actor'].forward(states).to(networks['actor'].device)
        noise = T.tensor(np.random.normal(scale=self.noise, size=mus.shape),
                         dtype=T.float).to(networks['actor'].device)
        mus_prime = T.clamp(mus + noise, -networks['actor'].min_max_action,
                            networks['actor'].min_max_action)
        self.time_step += 1
        return mus_prime.cpu().detach().numpy()

    def TD3_choose_action(self, observation, networks, n_actions):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise,
                                           size = (n_actions,)),
                          dtype=T.float).to(networks['actor'].device)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            mu = networks['actor'].forward(state).to(networks['actor'].device)
        mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(networks['actor'].device)
        mu_prime = T.clamp(mu_prime, -networks['actor'].min_max_action, networks['actor'].min_max_action)
        self.time_step += 1
        return mu_prime.unsqueeze(0).cpu().detach().numpy()
    
    def Attention_choose_action(self, observation, networks):
        return networks['attention'](observation).cpu().detach().numpy()

    
    def learn_DQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.arange(self.batch_size, device=networks['q_eval'].device)

        q_pred = networks['q_eval'](states)[indices, actions.long()]

        q_next = networks['q_next'](states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = self._q_target(rewards, q_next)

        loss = self._critic_loss_fn(q_target, q_pred)
        loss.backward()
        _check_nan(loss, f"DQN loss at step {networks['learn_step_counter']}")
        self._clip_critic_grad(networks['q_eval'])

        networks['q_eval'].optimizer.step()
        networks['learn_step_counter'] += 1

        self.decrement_epsilon()

        return loss.item()
    

    def learn_DDQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.arange(self.batch_size, device=networks['q_eval'].device)

        q_pred = networks['q_eval'](states)[indices, actions.long()]

        q_next = networks['q_next'](states_)
        q_eval = networks['q_eval'](states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = self._q_target(rewards, q_next[indices, max_actions])

        loss = self._critic_loss_fn(q_target, q_pred)

        loss.backward()
        _check_nan(loss, f"DDQN loss at step {networks['learn_step_counter']}")
        self._clip_critic_grad(networks['q_eval'])

        networks['q_eval'].optimizer.step()

        networks['learn_step_counter']+=1
        self._maybe_redo(networks['q_eval'], states, networks['learn_step_counter'])

        self.decrement_epsilon()

        return loss.item()

    def learn_DDQN_sf(self, networks):
        """Successor-Features DDQN learn step (GSP_SF_ENABLED).

        Two decoupled updates on the SF-DDQN pair (q_eval = DDQN_SF, q_next its
        target):

        1. psi TD update (the value computation itself):
             target_psi = phi_t + gamma * psi_next(s', a*)       (a* = argmax_a Q)
             loss_psi   = critic_loss( psi_eval(s_t, a_t), target_psi.detach() )
           applied per cumulant component (broadcast MSE/Huber over d_phi). The
           double-Q decoupling matches learn_DDQN: q_eval selects a*, q_next
           evaluates psi_next(s', a*). Because Q = psi . w, this trains the exact
           quantity that *is* the value — zeroing psi zeroes Q by construction.

        2. w reward-regression update (sets the value scale):
             pred_r = psi_eval(s_t, a_t).detach() . w
             loss_w = critic_loss( pred_r, r_target )
           r_target is the (reward-scaled) scalar reward (default) or the
           discounted-bootstrap reward-to-go (GSP_SF_W_TARGET='reward_to_go'). psi
           is detached here so the reward regression only moves w — it never
           perturbs the psi feature trunk, keeping the two objectives orthogonal.

        Stability: the same critic stabilizers as learn_DDQN apply to the Q=psi.w
        scale — reward_scale multiplies phi in the psi target AND the w target;
        q_target_clip bounds the psi bootstrap target; grad_clip_norm clips the psi
        trunk; critic_loss (huber) caps the per-sample gradient. w is init 1/d_phi
        so the initial Q is a bounded mean of psi's components.

        Returns:
            dict of diagnostics: sf_psi_loss, sf_w_loss, sf_psi_norm, sf_w_norm,
                sf_q_mean, sf_q_abs_max, w (list). None if the batch is unavailable.
        """
        q_eval = networks['q_eval']
        q_next = networks['q_next']
        device = q_eval.device

        result = networks['replay'].sample_buffer_sf(self.batch_size)
        states_np, actions_np, rewards_np, states_np_, dones_np, phi_np = result

        states = T.as_tensor(states_np, dtype=T.float32).to(device)
        actions = T.as_tensor(np.asarray(actions_np, dtype=np.float32)).to(device)
        rewards = T.as_tensor(rewards_np, dtype=T.float32).to(device)
        states_ = T.as_tensor(states_np_, dtype=T.float32).to(device)
        dones = T.as_tensor(dones_np).to(device)
        phi = T.as_tensor(phi_np, dtype=T.float32).to(device)  # (batch, d_phi)

        indices = T.arange(self.batch_size, device=device)
        act_idx = actions.long()

        # --- 1. psi TD update -------------------------------------------------
        q_eval.psi_optimizer.zero_grad()

        psi_all = q_eval.psi(states)                     # (batch, n_act, d_phi)
        psi_pred = psi_all[indices, act_idx]             # (batch, d_phi)

        with T.no_grad():
            # Double-Q: a* from q_eval, psi evaluated on q_next.
            q_eval_next = q_eval.forward(states_)        # (batch, n_act)
            max_actions = T.argmax(q_eval_next, dim=1)
            psi_next_all = q_next.psi(states_)           # (batch, n_act, d_phi)
            psi_next = psi_next_all[indices, max_actions]  # (batch, d_phi)
            psi_next[dones] = 0.0
            # reward_scale multiplies the immediate cumulant, mirroring _q_target.
            phi_scaled = phi if self.reward_scale == 1.0 else self.reward_scale * phi
            psi_target = phi_scaled + self.gamma * psi_next
            if self.q_target_clip > 0:
                psi_target = T.clamp(psi_target, -self.q_target_clip, self.q_target_clip)

        psi_loss = self._critic_loss_fn(psi_pred, psi_target)
        psi_loss.backward()
        _check_nan(psi_loss, f"SF psi loss at step {networks['learn_step_counter']}")
        self._clip_critic_grad(q_eval)
        q_eval.psi_optimizer.step()

        # --- 2. w reward-regression update ------------------------------------
        q_eval.w_optimizer.zero_grad()

        psi_detached = psi_pred.detach()                 # (batch, d_phi)
        pred_r = T.matmul(psi_detached, q_eval.w)        # (batch,)
        if self.gsp_sf_w_target == 'reward_to_go':
            # Regress Q = psi.w to the discounted-bootstrap target (value scale).
            with T.no_grad():
                q_next_boot = q_next.forward(states_)[indices, max_actions]
                q_next_boot = q_next_boot.clone()
                q_next_boot[dones] = 0.0
                r_target = self._q_target(rewards, q_next_boot)
        else:
            # Default: one-step w.phi ~= r (reward_scale-consistent with the target).
            r_target = rewards if self.reward_scale == 1.0 else self.reward_scale * rewards
        w_loss = self._critic_loss_fn(pred_r, r_target)
        w_loss.backward()
        _check_nan(w_loss, f"SF w loss at step {networks['learn_step_counter']}")
        q_eval.w_optimizer.step()

        networks['learn_step_counter'] += 1
        self._maybe_redo(q_eval, states, networks['learn_step_counter'])
        self.decrement_epsilon()

        with T.no_grad():
            q_now = T.matmul(psi_pred.detach(), q_eval.w)
            diagnostics = {
                'sf_psi_loss': float(psi_loss.item()),
                'sf_w_loss': float(w_loss.item()),
                'sf_psi_norm': float(psi_pred.detach().norm(dim=1).mean().item()),
                'sf_w_norm': float(q_eval.w.detach().norm().item()),
                'sf_q_mean': float(q_now.mean().item()),
                'sf_q_abs_max': float(q_now.abs().max().item()),
                'w': [float(v) for v in q_eval.w.detach().cpu().numpy()],
            }
        return diagnostics

    def _actor_gsp_weight_diag(self, actor_net, k: int) -> dict:
        """Compute the actor-GSP-feature reliance diagnostic for ``actor_net``.

        Reads the first linear layer weight (``fc1.weight``, shape
        ``(hidden, N_obs + K)``) of the network that consumes the augmented
        ``[obs | gsp]`` state (the DDQN Q-net or the TD3 actor). ``N_obs`` is
        ``self.input_size`` (the raw egocentric obs width) and ``K`` is the GSP
        slot width (``gsp_network_output``, passed in as ``k``). Returns the
        three ``actor_gsp_*`` reliance metrics.

        Pure read under no_grad; on any structural mismatch (unexpected layer
        name / shape) returns NaNs rather than perturbing or crashing training —
        this is a diagnostic, it must never gate the learn step.
        """
        n_obs = int(getattr(self, 'input_size'))
        first = getattr(actor_net, 'fc1', None)
        weight = getattr(first, 'weight', None)
        if weight is None or weight.dim() != 2 or weight.shape[1] != n_obs + int(k):
            nan = float('nan')
            return {
                'actor_gsp_feature_weight_norm': nan,
                'actor_obs_weight_norm_mean': nan,
                'actor_gsp_weight_ratio': nan,
            }
        return actor_gsp_feature_weight_diag(weight, n_obs, int(k))

    def learn_DDQN_e2e(self, networks, gsp_networks):
        """End-to-end joint training of DDQN + GSP head.

        At each learn step:
        1. Sample 7-value batch from main replay (requires gsp_obs_size > 0).
        2. Re-run GSP head on gsp_obs WITH gradient to produce fresh prediction.
        3. Replace the stale GSP scalar in the stored state with the fresh value.
        4. Run DDQN on augmented state.
        5. Compute combined loss: ddqn_loss + lambda * MSE(fresh_gsp, label).
        6. Backward through both networks, clip GSP gradients, step both optimizers.

        The next-state Q-target uses STORED states_ as-is (no GSP re-run) wrapped
        in torch.no_grad() — stable targets are critical for DDQN convergence.

        Args:
            networks: Main DDQN networks dict (must contain 'q_eval', 'q_next',
                'replay', 'learning_scheme', 'learn_step_counter').
            gsp_networks: GSP networks dict (must contain 'actor', 'learning_scheme').

        Returns:
            dict with keys: ddqn_loss, gsp_mse_loss, total_loss, gsp_grad_norm,
                gsp_grad_norm_pre_clip, ddqn_grad_norm, gsp_input_grad,
                gsp_pred_mean, gsp_pred_std, gsp_label_mean, gsp_label_std.
        """
        e2e_lambda = self.gsp_e2e_lambda
        device = networks['q_eval'].device

        # --- 1. Sample 7 values directly from main replay ---
        result = networks['replay'].sample_buffer(self.batch_size)
        states_np, actions_np, rewards_np, states_np_, dones_np, gsp_obs_np, gsp_labels_np = result

        states = T.as_tensor(states_np, dtype=T.float32).to(device)
        actions = T.as_tensor(np.asarray(actions_np, dtype=np.float32)).to(device)
        rewards = T.as_tensor(rewards_np, dtype=T.float32).to(device)
        states_ = T.as_tensor(states_np_, dtype=T.float32).to(device)
        dones = T.as_tensor(dones_np).to(device)
        gsp_obs = T.as_tensor(gsp_obs_np, dtype=T.float32).to(device)
        gsp_labels = T.as_tensor(gsp_labels_np, dtype=T.float32).to(device)

        # --- Zero both optimizers before any forward pass ---
        networks['q_eval'].optimizer.zero_grad()
        gsp_networks['actor'].optimizer.zero_grad()

        # --- 2. Re-run GSP head WITH gradient ---
        gsp_pred = gsp_networks['actor'].forward(gsp_obs)
        # gsp_pred shape: (batch, 1) or (batch,) — normalize to (batch, 1)
        if gsp_pred.dim() == 1:
            gsp_pred = gsp_pred.unsqueeze(1)

        # --- 3. Replace stale GSP prediction slot in state ---
        # State layout: [env_obs(input_size), gsp_pred(K), optional_gk(...)]
        # self.input_size is the raw env obs dimensionality (e.g. 31); the GSP
        # slot is gsp_network_output (K) wide — K=1 for the legacy scalar target,
        # K>1 for the size-K trajectory target (delta_theta_traj). The splice must
        # remove exactly K stale columns and insert exactly K fresh ones; slicing
        # a fixed width-1 slot left K-1 stale columns in place and grew the
        # augmented width past the Q-net's input, crashing the forward with
        # `mat1 and mat2 shapes cannot be multiplied` (confirmed 2026-07-08).
        _gsp_slot = int(getattr(self, 'gsp_network_output', 1))
        # CRITICAL: the actor's GSP slot must use the SAME representation at
        # act-time and learn-time. At act/store-time, agent.make_agent_state
        # (RL-CT) writes the SCALAR (K==1) as np.degrees(pred/10) = pred *
        # (180/pi/10) (the historical delta_theta_1d scaling), but for the VECTOR
        # (K>1) path it concatenates the RAW prediction with no rescaling. This
        # learn path re-runs the head and must mirror that exactly: scale only the
        # scalar slot. Previously the RAW scalar was spliced, so the Q-net was
        # trained on a ~5.73x-smaller value than it saw when acting AND than the
        # stored next-state slot — an internally inconsistent Bellman update that
        # prevented the E2E actor from learning. Applying the scale to a K>1 slot
        # would re-introduce that same inconsistency vs the RAW vector path, so it
        # is gated on K==1. The head's supervised MSE below always uses the
        # un-rescaled pred — the label arrives in whatever units RL-CT's label
        # pipeline stored (raw physical units by default; meters ×
        # GSP_TRAJ_LABEL_SCALE for the metric trajectory kinds since
        # RL-CT#35). No splice-side rescaling happens either way.
        if _gsp_slot == 1:
            _GSP_ACTOR_SCALE = float(np.degrees(1.0) / 10.0)  # == degrees(x/10)/x
            gsp_pred_actor = gsp_pred * _GSP_ACTOR_SCALE
        else:
            _GSP_ACTOR_SCALE = 1.0
            gsp_pred_actor = gsp_pred
        # GSP_E2E_STOP_GRAD_FEATURE: sever the actor's TD gradient at the splice so
        # it cannot perturb the head. The supervised MSE below still uses the RAW,
        # un-detached gsp_pred, so the head keeps learning to predict.
        if self.gsp_e2e_stop_grad_feature:
            gsp_pred_actor = gsp_pred_actor.detach()
        # GSP_E2E_NORMALIZE_FEATURE (opt-in): standardize the spliced feature to
        # ~unit variance so it is on the scale of the O(1) egocentric obs. This is
        # the SAME shared RunningStandardizer the acting splice
        # (agent.make_agent_state) reads — identical stats on both sides. Standardize
        # with the CURRENT (frozen) stats first, THEN fold this batch in, so the
        # value the critic sees here matches what acting saw with the pre-batch
        # stats (BatchNorm-style running estimate). Update uses the pre-standardized
        # slot representation (post-scale, matching the acting slot). Grad flows
        # through standardize (mean/std enter as constants). None → no-op.
        # Diagnostic: std of the spliced feature BEFORE / AFTER the NORMALIZE_FEATURE
        # standardization, to confirm the standardizer is actually rescaling the
        # feature. Read-only, under no_grad — never enters the loss. When normalize
        # is off (gsp_feature_stats is None) post == pre by construction.
        with T.no_grad():
            gsp_feature_std_prenorm = float(gsp_pred_actor.detach().std().item())
        if self.gsp_feature_stats is not None:
            gsp_pred_actor = self.gsp_feature_stats.standardize(gsp_pred_actor)
            self.gsp_feature_stats.update(gsp_pred.detach() * _GSP_ACTOR_SCALE)
        # GSP_E2E_SPLICE_GAIN: fixed constant salience gain, the LAST transform
        # before the splice (mirrors the acting splice in RL-CT
        # make_agent_state). 1.0 = exact no-op. The postnorm diagnostic below
        # deliberately reads AFTER the gain — it logs the scale the actor sees.
        if self.gsp_e2e_splice_gain != 1.0:
            gsp_pred_actor = gsp_pred_actor * self.gsp_e2e_splice_gain
        with T.no_grad():
            gsp_feature_std_postnorm = float(gsp_pred_actor.detach().std().item())
        gsp_idx = self.input_size
        augmented = T.cat(
            [states[:, :gsp_idx], gsp_pred_actor, states[:, gsp_idx + _gsp_slot:]],
            dim=1,
        )
        # retain_grad is only valid (and the gsp_input_grad diagnostic only
        # meaningful) when the actor-side feature carries grad. Under
        # GSP_E2E_STOP_GRAD_FEATURE the spliced pred is detached, so `augmented`
        # is a no-grad leaf and retain_grad would raise.
        if augmented.requires_grad:
            augmented.retain_grad()

        # --- 4. DDQN forward on augmented state ---
        indices = T.arange(self.batch_size, device=networks['q_eval'].device)
        q_pred = networks['q_eval'](augmented)[indices, actions.long()]

        # --- 5. Target Q using STORED next-state (no GSP re-run) ---
        with T.no_grad():
            q_next = networks['q_next'](states_)
            q_eval_next = networks['q_eval'](states_)
            max_actions = T.argmax(q_eval_next, dim=1)
            q_next[dones] = 0.0
            if self.gsp_e2e_unified_target_arith:
                # GSP_E2E_UNIFIED_TARGET_ARITH: identical Bellman arithmetic
                # to learn_DDQN — reward_scale * rewards + gamma * bootstrap,
                # then the Q_TARGET_CLIP clamp (_q_target). Closes the
                # IC-vs-E2E 10x effective-reward-scale asymmetry under
                # REWARD_SCALE recipes (see the flag's config-block comment).
                q_target = self._q_target(rewards, q_next[indices, max_actions])
            else:
                # Legacy (pre-flag) arithmetic, byte-identical and
                # golden-tested: raw rewards, no reward_scale, no target clip.
                q_target = rewards + self.gamma * q_next[indices, max_actions]

        # --- 6. Combined loss ---
        ddqn_loss = networks['q_eval'].loss(q_target, q_pred).to(device)

        if gsp_labels.dim() == gsp_pred.dim() - 1:
            gsp_labels = gsp_labels.unsqueeze(-1)
        else:
            gsp_labels = gsp_labels.view_as(gsp_pred)
        gsp_mse_loss = F.mse_loss(gsp_pred, gsp_labels)

        total_loss = ddqn_loss + e2e_lambda * gsp_mse_loss

        # --- 7. Backward + gradient clipping ---
        total_loss.backward()
        _check_nan(total_loss, f"E2E total loss at step {networks['learn_step_counter']}")

        # GSP_E2E_UNIFIED_TARGET_ARITH: mirror learn_DDQN's critic-grad-clip
        # treatment — _clip_critic_grad(q_eval) after backward, before the
        # optimizer step (a no-op when GRAD_CLIP_NORM=0, same as the plain
        # path). Legacy path: q_eval is NEVER clipped here (only the GSP head
        # clips below) — kept byte-identical. The ddqn_grad_norm diagnostic
        # below therefore reads the POST-clip norm when the flag is on: the
        # value the optimizer actually consumes (consumption-boundary
        # logging), consistent with the gsp post-clip norm next to it.
        if self.gsp_e2e_unified_target_arith:
            self._clip_critic_grad(networks['q_eval'])

        # Pre-clip GSP grad norm for diagnostics
        gsp_params = list(gsp_networks['actor'].parameters())
        gsp_grad_norm_pre_clip = float(
            T.norm(T.stack([p.grad.norm() for p in gsp_params if p.grad is not None]))
        )

        T.nn.utils.clip_grad_norm_(gsp_networks['actor'].parameters(), max_norm=1.0)

        # Post-clip GSP grad norm
        gsp_grad_norm = float(
            T.norm(T.stack([p.grad.norm() for p in gsp_params if p.grad is not None]))
        )

        # DDQN Q-eval grad norm (before step)
        q_params = list(networks['q_eval'].parameters())
        ddqn_grad_norm = float(
            T.norm(T.stack([p.grad.norm() for p in q_params if p.grad is not None]))
        )

        # Gradient at the GSP input dimension of the augmented state
        gsp_input_grad = None
        if augmented.grad is not None:
            gsp_input_grad = float(augmented.grad[:, gsp_idx].abs().mean().item())

        # --- 8. Step both optimizers ---
        networks['q_eval'].optimizer.step()
        gsp_networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1
        self.decrement_epsilon()

        # Actor GSP-feature reliance diagnostic (the headline causal-usage metric).
        # The augmented state feeds the DDQN Q-net, whose first layer fc1.weight has
        # shape (hidden, input_size + K). The GSP feature occupies the LAST K columns.
        # Pure read of the post-step weights under no_grad — no autograd edge, no
        # perturbation of the net. See feature_stats.actor_gsp_feature_weight_diag.
        weight_diag = self._actor_gsp_weight_diag(networks['q_eval'], _gsp_slot)

        return {
            'ddqn_loss': ddqn_loss.item(),
            'gsp_mse_loss': gsp_mse_loss.item(),
            'total_loss': total_loss.item(),
            'gsp_grad_norm': gsp_grad_norm,
            'gsp_grad_norm_pre_clip': gsp_grad_norm_pre_clip,
            'ddqn_grad_norm': ddqn_grad_norm,
            'gsp_input_grad': gsp_input_grad,
            'gsp_pred_mean': float(gsp_pred.detach().mean().item()),
            'gsp_pred_std': float(gsp_pred.detach().std().item()),
            'gsp_label_mean': float(gsp_labels.detach().mean().item()),
            'gsp_label_std': float(gsp_labels.detach().std().item()),
            'gsp_feature_std_prenorm': gsp_feature_std_prenorm,
            'gsp_feature_std_postnorm': gsp_feature_std_postnorm,
            **weight_diag,
        }

    def learn_DDQN_jepa_coupled(self, networks, gsp_networks):
        """Coupled-JEPA DDQN learn step (GSP_JEPA_COUPLE_VALUE).

        The literature-convergent fix for the causally-inert GSP prediction
        (docs/research/2026-07-05-literature-synthesis-causal-prediction.md).
        Structurally mirrors ``learn_DDQN_e2e`` but operates on the JEPA *latent*
        slot instead of a scalar Δθ prediction, and adds the JEPA self-prediction
        (latent) loss so the encoder is shaped by BOTH signals:

          * value path:  the DDQN TD loss gradient flows into gsp_encoder_online
            because we re-encode the stored raw gsp_obs WITH gradient and splice
            the FRESH latent into the augmented state before the Q-forward
            (Ni et al. 2401.08898 Thm 2 — value + self-prediction ⇒ reward-
            predictive; ZP alone stays decision-irrelevant).
          * self-prediction path: predictor(z_t[, a_t]) vs EMA-target(z_{t+k}),
            with stop-grad on the target encoder (the standard JEPA collapse
            guard). This term is the same objective as learn_gsp_jepa; here it is
            trained jointly with the value loss on the SAME encoder.

        Ni-couple vs Dreamer-freeze resolution (GSP_JEPA_VALUE_STOPGRAD_ACTOR):
        DDQN has no separate actor — the Q-net is value AND policy — so the
        spliced latent is simultaneously the value representation and the
        actor input. When the flag is True the spliced latent is detached before
        the Q-net, so the value loss cannot rewrite the encoder (pure Dreamer-
        freeze; encoder shaped only by self-prediction). When False (default for
        the coupling arm) the value gradient flows fully into the encoder.

        Requires the main replay to carry gsp_obs (built when GSP_JEPA_COUPLE_VALUE
        is set). Returns a diagnostics dict including latent_rank / latent_var /
        jepa_pred_mse so the same metrics the uncoupled path logs are available,
        plus the value-loss / total-loss / grad-norm terms.
        """
        if networks['replay'].mem_ctr < self.batch_size:
            return None

        device = networks['q_eval'].device
        enc_dim = int(self.gsp_encoder_dim)
        # Latent slot start index in the stored augmented state. Concat design:
        # [env_obs | latent | (global)] → slot begins right after raw env obs.
        # Latent-primary (GSP_ACTOR_LATENT_PRIMARY): env_obs is dropped from the
        # actor's Q-net input, so the augmented state is [latent | (global)] and
        # the slot begins at index 0. Must match network_input_size (actor.py) and
        # make_agent_state (RL-CollectiveTransport) or the state_dict sizes / the
        # splice would desync.
        gsp_idx = 0 if getattr(self, 'gsp_actor_latent_primary', False) else self.input_size

        # The JEPA encoder/predictor default to their own device (cuda:0 or cpu),
        # which matches the Q-net in production (single CUDA device) but can differ
        # on MPS/multi-device hosts. Co-locate them with the Q-net so the spliced
        # latent, the Q-forward, and the self-prediction loss all share one device.
        if getattr(self.gsp_encoder_online, 'device', device) != device:
            self.gsp_encoder_online.to(device)
            self.gsp_encoder_online.device = device
            self.gsp_encoder_target.to(device)
            self.gsp_encoder_target.device = device
            self.gsp_predictor.to(device)
            self.gsp_predictor.device = device
        value_coef = self.gsp_jepa_value_coef
        selfpred_coef = self.gsp_jepa_selfpred_coef
        stopgrad_actor = self.gsp_jepa_value_stopgrad_actor

        # --- 1. Sample 7-value batch from main replay ---
        result = networks['replay'].sample_buffer(self.batch_size)
        states_np, actions_np, rewards_np, states_np_, dones_np, gsp_obs_np, _ = result
        states = T.as_tensor(states_np, dtype=T.float32).to(device)
        actions = T.as_tensor(np.asarray(actions_np, dtype=np.float32)).to(device)
        rewards = T.as_tensor(rewards_np, dtype=T.float32).to(device)
        states_ = T.as_tensor(states_np_, dtype=T.float32).to(device)
        dones = T.as_tensor(dones_np).to(device)
        gsp_obs = T.as_tensor(gsp_obs_np, dtype=T.float32).to(device)

        networks['q_eval'].optimizer.zero_grad()
        self._jepa_online_optimizer.zero_grad()

        # --- 2. Re-encode raw gsp_obs WITH gradient → fresh latent ---
        z_t = self.gsp_encoder_online(gsp_obs)  # (batch, enc_dim), grad-tracked

        # --- 3. Splice the fresh latent into the augmented state ---
        # Stop-grad the latent when used as the actor/Q-net INPUT if the
        # Dreamer-freeze resolution is selected; otherwise the value gradient
        # flows through into the encoder (Ni coupling).
        latent_for_value = z_t.detach() if stopgrad_actor else z_t
        augmented = T.cat(
            [states[:, :gsp_idx], latent_for_value, states[:, gsp_idx + enc_dim:]],
            dim=1,
        )

        # --- 4. DDQN forward on augmented state ---
        indices = T.arange(self.batch_size, device=networks['q_eval'].device)
        q_pred = networks['q_eval'](augmented)[indices, actions.long()]

        # --- 5. Stable Q-target from stored next-state (no encoder re-run) ---
        with T.no_grad():
            q_next = networks['q_next'](states_)
            q_eval_next = networks['q_eval'](states_)
            max_actions = T.argmax(q_eval_next, dim=1)
            q_next[dones] = 0.0
            bootstrap = q_next[indices, max_actions]
            q_target = self._q_target(rewards, bootstrap)

        ddqn_loss = self._critic_loss_fn(q_target, q_pred).to(device)

        # --- 6. JEPA self-prediction loss (couples z_t to its own future) ---
        if self.gsp_jepa_action_cond and self.gsp_predictor.action_dim > 0:
            # One-hot the discrete DDQN action to the configured action width.
            a_dim = self.gsp_predictor.action_dim
            a_idx = actions.long().to(device)
            a_onehot = F.one_hot(a_idx.clamp(0, a_dim - 1), num_classes=a_dim).float()
            z_pred = self.gsp_predictor(z_t, a_onehot)
        else:
            z_pred = self.gsp_predictor(z_t)

        # SimNorm consistency: z_t and z_target already live on the simplex (the
        # encoder applies SimNorm inside forward). The predictor output does NOT,
        # so project it onto the same simplex before the self-prediction MSE so
        # both operands are comparable (target/online consistent). No-op when the
        # flag is off.
        if self.gsp_jepa_simnorm:
            z_pred = simnorm(z_pred, self.gsp_jepa_simnorm_group_size)

        # Target latent from the EMA target encoder on the raw gsp_obs, detached.
        # (state_{t+k} is not co-indexed in the MAIN replay; the coupled step
        # trains the predictor as a same-input consistency objective against the
        # slow target encoder — the self-prediction collapse guard — while the
        # value path provides the decision-relevance. The dedicated (t, t+k)
        # temporal prediction continues in learn_gsp_jepa on the JEPA replay.)
        with T.no_grad():
            z_target = self.gsp_encoder_target(gsp_obs)

        if self.gsp_jepa_cosine_loss:
            loss_selfpred = jepa_cosine_loss(z_pred, z_target)
        else:
            loss_selfpred = F.mse_loss(z_pred, z_target)

        # --- Optional VICReg anti-collapse on the online latent ---
        vicreg_term = T.zeros((), device=z_t.device)
        if self.gsp_vicreg_enabled:
            vicreg_term = (
                self.gsp_vicreg_var_coef * vicreg_variance_loss(z_t, target_std=1.0)
                + self.gsp_vicreg_cov_coef * vicreg_covariance_loss(z_t)
            )

        total_loss = (
            value_coef * ddqn_loss
            + selfpred_coef * loss_selfpred
            + vicreg_term
        )

        # --- 7. Joint backward + optimizer steps ---
        total_loss.backward()
        _check_nan(total_loss, f"coupled-JEPA loss at step {networks['learn_step_counter']}")

        enc_params = list(self.gsp_encoder_online.parameters())
        enc_grad_norm = float(
            T.norm(T.stack([p.grad.norm() for p in enc_params if p.grad is not None]))
        ) if any(p.grad is not None for p in enc_params) else 0.0
        if self.grad_clip_norm > 0:
            T.nn.utils.clip_grad_norm_(
                list(self.gsp_encoder_online.parameters())
                + list(self.gsp_predictor.parameters()),
                max_norm=self.grad_clip_norm,
            )
            T.nn.utils.clip_grad_norm_(
                networks['q_eval'].parameters(), max_norm=self.grad_clip_norm
            )

        networks['q_eval'].optimizer.step()
        self._jepa_online_optimizer.step()

        # --- 8. EMA update of the target encoder ---
        self._update_jepa_target_encoder(self.gsp_encoder_ema_tau)

        networks['learn_step_counter'] += 1
        self.decrement_epsilon()

        # --- 9. Latent diagnostics (mirror learn_gsp_jepa) ---
        with T.no_grad():
            latent_var = float(z_t.var(dim=0).mean().item())
            z_cpu = z_t.detach().cpu()
            try:
                sv = T.linalg.svdvals(z_cpu)
                latent_rank = float((sv > sv[0] * 0.01).sum().item())
            except Exception:
                latent_rank = float("nan")

        stats = {
            'ddqn_loss': float(ddqn_loss.item()),
            'jepa_pred_mse': float(loss_selfpred.item()),
            'total_loss': float(total_loss.item()),
            'latent_var': latent_var,
            'latent_rank': latent_rank,
            'encoder_grad_norm': enc_grad_norm,
        }
        # Surface stats to Main.py's JEPA recorder in the same shape it expects.
        self.last_gsp_jepa_stats = {
            'var': latent_var,
            'rank': latent_rank,
            'pred_mse': float(loss_selfpred.item()),
        }
        return stats

    def learn_DDPG(self, networks, gsp = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)
        target_actions = networks['target_actor'](states_)
        q_value_ = networks['target_critic'](states_, target_actions)

        target = self._q_target(T.unsqueeze(rewards, 1), q_value_)

        #Critic Update
        networks['critic'].optimizer.zero_grad()

        q_value = networks['critic'](states, actions)
        value_loss = self._critic_loss_fn(q_value, target)
        value_loss.backward()
        _check_nan(value_loss, f"DDPG critic loss at step {networks['learn_step_counter']}")
        self._clip_critic_grad(networks['critic'])
        networks['critic'].optimizer.step()

        #Actor Update
        networks['actor'].optimizer.zero_grad()

        new_policy_actions = networks['actor'](states)
        actor_loss = -networks['critic'](states, new_policy_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        _check_nan(actor_loss, f"DDPG actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()
    
    def learn_RDDPG(self, networks, gsp = False, recurrent = False):
        mem_result = self.sample_memory(networks)
        if len(mem_result) == 7:
            states, actions, rewards, states_, dones, h_batch, c_batch = mem_result
            device = networks['actor'].device
            h_0 = T.tensor(np.array(h_batch), dtype=T.float32).to(device)
            c_0 = T.tensor(np.array(c_batch), dtype=T.float32).to(device)
            # h_batch shape: (batch, num_layers, 1, hidden) -> (num_layers, batch, hidden)
            h_0 = h_0.squeeze(2).permute(1, 0, 2).contiguous()
            c_0 = c_0.squeeze(2).permute(1, 0, 2).contiguous()
            hidden_init = (h_0, c_0)
        else:
            states, actions, rewards, states_, dones = mem_result
            hidden_init = None

        # states: (batch, seq_len, obs_dim)
        # actions: (batch, seq_len, act_dim)
        seq_len = states.shape[1]
        burn_in_len = seq_len // 2
        train_len = seq_len - burn_in_len

        # Split into burn-in and training portions
        burn_states = states[:, :burn_in_len, :]
        train_states = states[:, burn_in_len:, :]
        burn_states_ = states_[:, :burn_in_len, :]
        train_states_ = states_[:, burn_in_len:, :]
        train_actions = actions[:, burn_in_len:, :]
        train_rewards = rewards[:, burn_in_len:]

        # Burn-in: refresh hidden state without gradients
        with T.no_grad():
            if burn_in_len > 0:
                # Run burn-in through actor encoder to get hidden state
                _, actor_hidden = networks['actor'].ee(burn_states, hidden=hidden_init)
                _, critic_hidden = networks['critic'].ee(burn_states, hidden=hidden_init)
                _, target_actor_hidden = networks['target_actor'].ee(burn_states_, hidden=hidden_init)
                _, target_critic_hidden = networks['target_critic'].ee(burn_states_, hidden=hidden_init)
            else:
                actor_hidden = hidden_init
                critic_hidden = hidden_init
                target_actor_hidden = hidden_init
                target_critic_hidden = hidden_init

        # Detach hidden states so burn-in gradients don't flow
        if actor_hidden is not None:
            actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
            critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())
            target_actor_hidden = (target_actor_hidden[0].detach(), target_actor_hidden[1].detach())
            target_critic_hidden = (target_critic_hidden[0].detach(), target_critic_hidden[1].detach())

        # Target computation (no gradients)
        with T.no_grad():
            target_actions, _ = networks['target_actor'](train_states_, hidden=target_actor_hidden)
            q_value_, _ = networks['target_critic'](train_states_, target_actions, hidden=target_critic_hidden)
            # Use last timestep for Bellman target
            q_last_ = q_value_[:, -1, :]  # (batch, 1)
            r_last = train_rewards[:, -1]  # (batch,)
            target = self._q_target(r_last.unsqueeze(1), q_last_)  # (batch, 1)

        # Critic update
        networks['critic'].optimizer.zero_grad()
        q_value, _ = networks['critic'](train_states, train_actions, hidden=critic_hidden)
        q_last = q_value[:, -1, :]  # (batch, 1)
        value_loss = self._critic_loss_fn(q_last, target)
        value_loss.backward()
        _check_nan(value_loss, f"RDDPG critic loss at step {networks['learn_step_counter']}")
        self._clip_critic_grad(networks['critic'])
        networks['critic'].optimizer.step()

        # Actor update
        networks['actor'].optimizer.zero_grad()
        new_policy_actions, _ = networks['actor'](train_states, hidden=actor_hidden)
        # Re-run critic with fresh hidden (detached) for actor loss
        actor_q_val, _ = networks['critic'](train_states, new_policy_actions, hidden=critic_hidden)
        actor_loss = -actor_q_val[:, -1, :].mean()
        actor_loss.backward()
        _check_nan(actor_loss, f"RDDPG actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()

    def learn_TD3(self, networks, gsp = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)

        with T.no_grad():
            target_actions = networks['target_actor'].forward(states_)
            noise = T.clamp(
                T.tensor(np.random.normal(0, 0.2, size=target_actions.shape).astype(np.float32)),
                -0.5, 0.5
            ).to(target_actions.device)
            target_actions = T.clamp(target_actions + noise, -self.min_max_action, self.min_max_action)

            q1_ = networks['target_critic_1'].forward(states_, target_actions)
            q2_ = networks['target_critic_2'].forward(states_, target_actions)

            q1_[dones] = 0.0
            q2_[dones] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_)
            target = self._q_target(rewards, critic_value_)

        q1 = networks['critic_1'].forward(states, actions).squeeze()
        q2 = networks['critic_2'].forward(states, actions).squeeze()

        networks['critic_1'].optimizer.zero_grad()
        networks['critic_2'].optimizer.zero_grad()

        q1_loss = self._critic_loss_fn(target, q1)
        q2_loss = self._critic_loss_fn(target, q2)
        critic_loss = q1_loss + q2_loss

        critic_loss.backward()
        _check_nan(critic_loss, f"TD3 critic loss at step {networks['learn_step_counter']}")
        self._clip_critic_grad(networks['critic_1'], networks['critic_2'])
        networks['critic_1'].optimizer.step()
        networks['critic_2'].optimizer.step()

        networks['learn_step_counter'] += 1

        if networks['learn_step_counter'] % self.update_actor_iter != 0:
            return 0, 0

        networks['actor'].optimizer.zero_grad()
        actor_q1_loss = networks['critic_1'].forward(states, networks['actor'].forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        _check_nan(actor_loss, f"TD3 actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        self.update_TD3_network_parameters(self.tau, networks)

        return actor_loss.item()

    def learn_TD3_e2e(self, networks, gsp_networks):
        """End-to-end joint training of TD3 + GSP head (cross-head charter arm).

        The continuous-actor mirror of ``learn_DDQN_e2e``. At each learn step the
        GSP head is re-run WITH gradient on the stored gsp_obs, its (scaled,
        optionally detached) prediction is spliced into the actor/critic CURRENT
        state at index ``self.input_size`` (feature #32), and the head's own
        supervised MSE loss is folded into the critic backward so the head trains
        every learn step. The TD3 twin-critic update, target-policy smoothing, and
        delayed actor update all follow ``learn_TD3``.

        Current-vs-next state (mirrors ``learn_DDQN_e2e``):
          * CURRENT state (feeds critic_1/critic_2 for the TD loss and the online
            actor for the delayed policy loss): the stale stored GSP scalar is
            REPLACED with the freshly re-run head prediction (scaled by
            ``_GSP_ACTOR_SCALE`` = degrees(x/10)/x, detached iff
            ``gsp_e2e_stop_grad_feature``). This is the only place the head's TD
            gradient can enter the value networks.
          * NEXT state (``states_``): used AS-STORED under ``T.no_grad()`` for the
            target-actor + target-critics, exactly like ``learn_DDQN_e2e`` uses the
            stored next-state. The stored ``states_`` already contains the scaled
            feature #32 because RL-CT ``agent.make_agent_state`` writes
            ``degrees(pred/10)`` into every acted (and therefore stored) state, so
            current, next, and inference agree without re-running the head on the
            next state.

        The GSP head's MSE loss (on the RAW, un-detached prediction vs the label)
        is added to the CRITIC loss so it is applied on EVERY learn step — the TD3
        actor update is delayed (every ``update_actor_iter`` steps) and must not
        gate the head's supervised training.

        Returns a diagnostics dict shaped like ``learn_TD3``'s scalar losses plus
        the GSP diagnostics ``learn_DDQN_e2e`` emits (so the h5 e2e logger captures
        them). ``actor_loss`` is None on non-actor-update steps.
        """
        e2e_lambda = self.gsp_e2e_lambda
        device = networks['critic_1'].device

        # --- 1. Sample 7 values directly from main replay ---
        result = networks['replay'].sample_buffer(self.batch_size)
        states_np, actions_np, rewards_np, states_np_, dones_np, gsp_obs_np, gsp_labels_np = result

        states = T.as_tensor(states_np, dtype=T.float32).to(device)
        actions = T.as_tensor(np.asarray(actions_np, dtype=np.float32)).to(device)
        rewards = T.as_tensor(rewards_np, dtype=T.float32).to(device)
        states_ = T.as_tensor(states_np_, dtype=T.float32).to(device)
        dones = T.as_tensor(dones_np).to(device)
        gsp_obs = T.as_tensor(gsp_obs_np, dtype=T.float32).to(device)
        gsp_labels = T.as_tensor(gsp_labels_np, dtype=T.float32).to(device)

        # --- 2. Re-run GSP head WITH gradient ---
        gsp_networks['actor'].optimizer.zero_grad()
        gsp_pred = gsp_networks['actor'].forward(gsp_obs)
        if gsp_pred.dim() == 1:
            gsp_pred = gsp_pred.unsqueeze(1)

        # --- 3. Splice the fresh (scaled) GSP prediction into the CURRENT state ---
        # Scaling, K-slot width, and stop-grad semantics identical to
        # learn_DDQN_e2e: the GSP slot is gsp_network_output (K) wide; splice
        # removes exactly K stale columns and inserts K fresh ones. The scalar
        # (K==1) slot uses degrees(pred/10) to match make_agent_state at act/store
        # time; the vector (K>1) path is RAW (make_agent_state does not rescale
        # vectors). The head's supervised MSE below still uses the RAW pred.
        _gsp_slot = int(getattr(self, 'gsp_network_output', 1))
        if _gsp_slot == 1:
            _GSP_ACTOR_SCALE = float(np.degrees(1.0) / 10.0)  # == degrees(x/10)/x
            gsp_pred_actor = gsp_pred * _GSP_ACTOR_SCALE
        else:
            _GSP_ACTOR_SCALE = 1.0
            gsp_pred_actor = gsp_pred
        if self.gsp_e2e_stop_grad_feature:
            gsp_pred_actor = gsp_pred_actor.detach()
        # GSP_E2E_NORMALIZE_FEATURE (opt-in): standardize the spliced feature to
        # ~unit variance using the SAME shared RunningStandardizer the acting splice
        # (agent.make_agent_state) reads. Standardize with the current (frozen)
        # stats, then fold this batch in (BatchNorm-style running estimate). Update
        # uses the post-scale slot representation, matching the acting slot. Grad
        # flows through standardize (mean/std are constants). None → no-op.
        # Diagnostic: std of the spliced feature BEFORE / AFTER standardization
        # (confirms the standardizer rescales it). Read-only, under no_grad; when
        # normalize is off post == pre by construction. Mirrors learn_DDQN_e2e.
        with T.no_grad():
            gsp_feature_std_prenorm = float(gsp_pred_actor.detach().std().item())
        if self.gsp_feature_stats is not None:
            gsp_pred_actor = self.gsp_feature_stats.standardize(gsp_pred_actor)
            self.gsp_feature_stats.update(gsp_pred.detach() * _GSP_ACTOR_SCALE)
        # GSP_E2E_SPLICE_GAIN: fixed constant salience gain, the LAST transform
        # before the splice (mirrors the acting splice in RL-CT
        # make_agent_state). 1.0 = exact no-op. The postnorm diagnostic below
        # deliberately reads AFTER the gain — it logs the scale the actor sees.
        if self.gsp_e2e_splice_gain != 1.0:
            gsp_pred_actor = gsp_pred_actor * self.gsp_e2e_splice_gain
        with T.no_grad():
            gsp_feature_std_postnorm = float(gsp_pred_actor.detach().std().item())
        gsp_idx = self.input_size
        augmented = T.cat(
            [states[:, :gsp_idx], gsp_pred_actor, states[:, gsp_idx + _gsp_slot:]],
            dim=1,
        )
        # retain_grad only valid (and gsp_input_grad only meaningful) when the
        # actor-side feature carries grad — under stop-grad `augmented` is a
        # no-grad leaf and retain_grad would raise.
        if augmented.requires_grad:
            augmented.retain_grad()

        # --- 4. TD3 target: STORED next-state, no GSP re-run, no_grad ---
        with T.no_grad():
            target_actions = networks['target_actor'].forward(states_)
            noise = T.clamp(
                T.tensor(np.random.normal(0, 0.2, size=target_actions.shape).astype(np.float32)),
                -0.5, 0.5
            ).to(target_actions.device)
            target_actions = T.clamp(
                target_actions + noise, -self.min_max_action, self.min_max_action
            )
            q1_ = networks['target_critic_1'].forward(states_, target_actions)
            q2_ = networks['target_critic_2'].forward(states_, target_actions)
            q1_[dones] = 0.0
            q2_[dones] = 0.0
            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)
            critic_value_ = T.min(q1_, q2_)
            target = self._q_target(rewards, critic_value_)

        # --- 5. Twin-critic forward on the AUGMENTED current state ---
        q1 = networks['critic_1'].forward(augmented, actions).squeeze()
        q2 = networks['critic_2'].forward(augmented, actions).squeeze()

        networks['critic_1'].optimizer.zero_grad()
        networks['critic_2'].optimizer.zero_grad()

        q1_loss = self._critic_loss_fn(target, q1)
        q2_loss = self._critic_loss_fn(target, q2)
        critic_loss = q1_loss + q2_loss

        # --- 6. GSP head MSE on the RAW pred, applied EVERY learn step ---
        if gsp_labels.dim() == gsp_pred.dim() - 1:
            gsp_labels = gsp_labels.unsqueeze(-1)
        else:
            gsp_labels = gsp_labels.view_as(gsp_pred)
        gsp_mse_loss = F.mse_loss(gsp_pred, gsp_labels)

        combined_loss = critic_loss + e2e_lambda * gsp_mse_loss

        combined_loss.backward()
        _check_nan(combined_loss, f"TD3 e2e critic+gsp loss at step {networks['learn_step_counter']}")

        # Pre-clip GSP grad norm for diagnostics
        gsp_params = list(gsp_networks['actor'].parameters())
        gsp_grad_norm_pre_clip = float(
            T.norm(T.stack([p.grad.norm() for p in gsp_params if p.grad is not None]))
        )
        T.nn.utils.clip_grad_norm_(gsp_networks['actor'].parameters(), max_norm=1.0)
        gsp_grad_norm = float(
            T.norm(T.stack([p.grad.norm() for p in gsp_params if p.grad is not None]))
        )

        # Gradient at the GSP input dimension of the augmented state.
        gsp_input_grad = None
        if augmented.grad is not None:
            gsp_input_grad = float(augmented.grad[:, gsp_idx].abs().mean().item())

        self._clip_critic_grad(networks['critic_1'], networks['critic_2'])
        networks['critic_1'].optimizer.step()
        networks['critic_2'].optimizer.step()
        gsp_networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        # Actor GSP-feature reliance diagnostic (headline causal-usage metric).
        # In TD3 the spliced feature feeds the actor (and critics); the actor's
        # first layer fc1.weight is (hidden, input_size + K), GSP in the LAST K
        # columns. Pure read of the actor weights under no_grad — no perturbation.
        weight_diag = self._actor_gsp_weight_diag(networks['actor'], _gsp_slot)

        diagnostics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': None,
            'gsp_mse_loss': gsp_mse_loss.item(),
            'gsp_grad_norm': gsp_grad_norm,
            'gsp_grad_norm_pre_clip': gsp_grad_norm_pre_clip,
            'gsp_input_grad': gsp_input_grad,
            'gsp_pred_mean': float(gsp_pred.detach().mean().item()),
            'gsp_pred_std': float(gsp_pred.detach().std().item()),
            'gsp_label_mean': float(gsp_labels.detach().mean().item()),
            'gsp_label_std': float(gsp_labels.detach().std().item()),
            'gsp_feature_std_prenorm': gsp_feature_std_prenorm,
            'gsp_feature_std_postnorm': gsp_feature_std_postnorm,
            **weight_diag,
        }

        # --- 7. Delayed actor update (mirrors learn_TD3) ---
        if networks['learn_step_counter'] % self.update_actor_iter != 0:
            return diagnostics

        # Re-run the head so the actor's policy loss sees a fresh spliced feature
        # consistent with make_agent_state; scaled + optionally detached the same
        # way. The head's optimizer already stepped this iteration, so DO NOT add
        # this forward's MSE again — the actor loss trains the policy, and (when
        # the feature carries grad) shapes the head via the actor's deterministic
        # policy gradient exactly as the DDQN TD gradient does.
        gsp_pred_actor_step = gsp_networks['actor'].forward(gsp_obs)
        if gsp_pred_actor_step.dim() == 1:
            gsp_pred_actor_step = gsp_pred_actor_step.unsqueeze(1)
        gsp_pred_actor_step = gsp_pred_actor_step * _GSP_ACTOR_SCALE
        if self.gsp_e2e_stop_grad_feature:
            gsp_pred_actor_step = gsp_pred_actor_step.detach()
        # Mirror the critic-path slot pipeline exactly (standardize with the
        # frozen stats, then the fixed splice gain): the actor's deterministic
        # policy gradient must be computed on the SAME feature scale the critic
        # was trained on and acting sees, or the policy is optimized on a state
        # distribution it never encounters (review finding, 2026-07-10).
        if self.gsp_feature_stats is not None:
            gsp_pred_actor_step = self.gsp_feature_stats.standardize(
                gsp_pred_actor_step
            )
        if self.gsp_e2e_splice_gain != 1.0:
            gsp_pred_actor_step = gsp_pred_actor_step * self.gsp_e2e_splice_gain
        augmented_actor = T.cat(
            [states[:, :gsp_idx], gsp_pred_actor_step,
             states[:, gsp_idx + _gsp_slot:]],
            dim=1,
        )

        networks['actor'].optimizer.zero_grad()
        gsp_networks['actor'].optimizer.zero_grad()
        actor_q1_loss = networks['critic_1'].forward(
            augmented_actor, networks['actor'].forward(augmented_actor)
        )
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        _check_nan(actor_loss, f"TD3 e2e actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()
        # Let the actor's deterministic-policy gradient flow into the head too,
        # unless it was detached at the splice (stop-grad flag).
        if not self.gsp_e2e_stop_grad_feature:
            T.nn.utils.clip_grad_norm_(gsp_networks['actor'].parameters(), max_norm=1.0)
            gsp_networks['actor'].optimizer.step()

        self.update_TD3_network_parameters(self.tau, networks)

        diagnostics['actor_loss'] = actor_loss.item()
        return diagnostics

    def learn_attention(self, networks):
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return 0
        observations, labels = self.sample_attention_memory(networks)
        networks['learn_step_counter'] += 1
        networks['attention'].optimizer.zero_grad()
        pred_headings = networks['attention'](observations)
        loss = Loss(pred_headings, labels.unsqueeze(-1))
        loss.backward()
        _check_nan(loss, f"Attention loss at step {networks['learn_step_counter']}")
        networks['attention'].optimizer.step()
        return loss.item()

    def learn_gsp_mse(self, networks, recurrent: bool = False):
        """Train the GSP prediction network via direct supervised MSE.

        Replaces the DDPG/RDDPG actor-critic path for non-attention GSP variants.
        Samples (state, label) pairs from `networks['replay']`, forwards the state
        through the actor network, and minimizes MSE against the label. The label
        is stored in the action field of the replay buffer by convention — see
        RL-CollectiveTransport Main.py's store_gsp_transition call sites.

        Rationale: see docs/research/2026-04-13-gsp-information-collapse-analysis.md
        in the Stelaris repo. Training the GSP predictor as a DDPG actor-critic
        on a clipped negative-MSE reward produced an information-collapsed
        predictor whose output was worse than predicting the constant mean.
        Direct supervised MSE has a non-vanishing gradient `2(pred-label)` that
        drives the predictor toward the label regardless of how flat the reward
        landscape is.

        Task 5: Optional VICReg variance+covariance penalty on the penultimate
        feature vector (Bardes, Ponce, LeCun ICLR 2022). Guarded by
        self.gsp_vicreg_enabled (default False). When enabled, this ADDS two
        loss terms to the MSE loss targeting dimensional collapse of the
        encoder features — the failure mode that LayerNorm alone only
        partially addresses for MSE regression aux heads (Lyle 2024 +
        literature review 2026-04-15).

        Design notes (from the red-team audit):
        - target_std is scale-aware: defaults to running estimate of label std
          to avoid mismatched-scale saturation of the variance hinge
        - covariance coefficient normalized by feature dim
        - variance hinge: F.relu(target_std - pred_std).mean()
        """
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return None

        # Defensive: diagnostics put the actor in eval() and don't always
        # restore train(). cuDNN's RNN backward fails if the LSTM is in eval
        # mode at backward time ("cudnn RNN backward can only be called in
        # training mode") — Mac MPS doesn't have this restriction so the bug
        # is GPU-only. Restoring train() here is a no-op for non-recurrent
        # heads and prevents the crash for RDDPG (R-GSP-N).
        networks['actor'].train()

        vicreg_enabled = self.gsp_vicreg_enabled

        if recurrent:
            mem_result = self.sample_memory(networks)
            if len(mem_result) == 7:
                states, labels, _, _, _, _, _ = mem_result
            else:
                states, labels, _, _, _ = mem_result
            networks['actor'].optimizer.zero_grad()
            preds_out = networks['actor'](states, hidden=None)
            preds = preds_out[0] if isinstance(preds_out, tuple) else preds_out
            if preds.dim() == labels.dim() + 1:
                labels_shaped = labels.unsqueeze(-1)
            else:
                labels_shaped = labels.view_as(preds)
            mse_loss = F.mse_loss(preds, labels_shaped)
            # VICReg not yet supported for recurrent path (RDDPGActorNetwork
            # forward signature differs — would require a separate feature
            # extraction hook). Only apply to non-recurrent for now.
            loss = mse_loss
        else:
            states, labels, _, _, _ = self.sample_memory(networks)
            networks['actor'].optimizer.zero_grad()
            if vicreg_enabled:
                preds, features = networks['actor'].forward(states, return_features=True)
            else:
                preds = networks['actor'].forward(states)
                features = None
            # labels shape: (batch,) or (batch, 1). preds shape: (batch, 1).
            if labels.dim() == preds.dim() - 1:
                labels_shaped = labels.unsqueeze(-1)
            else:
                labels_shaped = labels.view_as(preds)
            mse_loss = F.mse_loss(preds, labels_shaped)
            loss = mse_loss

            if vicreg_enabled and features is not None:
                var_coef = self.gsp_vicreg_var_coef
                cov_coef = self.gsp_vicreg_cov_coef
                # Scale-aware target_std: match the batch's label std so the
                # variance hinge doesn't force features to saturate the
                # downstream tanh head. Clamp to >= 0.01 for numerical safety.
                with T.no_grad():
                    label_std = float(labels_shaped.std().clamp(min=0.01).item())
                var_loss = vicreg_variance_loss(features, target_std=label_std)
                cov_loss = vicreg_covariance_loss(features)
                loss = mse_loss + var_coef * var_loss + cov_coef * cov_loss

            # Phase 3 — L2-ER regularization.
            # L_total = MSE - lambda_er * sum(erank_per_layer)
            # Minimising L_total maximises effective rank at each layer, counteracting
            # dormancy / rank collapse. gsp_l2er_loss returns the positive erank_sum
            # so we subtract lambda * that quantity from the MSE loss.
            # Guard with hasattr so the path is robust against non-DDPG GSP heads
            # (attention variant does not have .fc1 / .fc2).
            l2er_lambda = self.gsp_l2er_lambda
            if l2er_lambda > 0.0 and hasattr(networks['actor'], 'fc1'):
                l2er_erank_sum = -gsp_l2er_loss(networks['actor'], states)
                # gsp_l2er_loss returns -(erank1+erank2); negating yields erank_sum > 0.
                loss = loss - l2er_lambda * l2er_erank_sum

        # Snapshot head outputs BEFORE the MSE backward+step.
        # We capture pre_outputs here (detached, no graph) so that after the
        # optimizer mutates the weights we can measure how much the function
        # changed on this same batch. This is C-CHAIN's "reference batch"
        # snapshot — Tang et al. 2506.00592 Eq. (3).
        cchain_lambda = self.gsp_cchain_lambda
        if cchain_lambda > 0.0 and not recurrent:
            with T.no_grad():
                pre_outputs = networks['actor'].forward(states).detach().clone()

        loss.backward()
        _check_nan(loss, f"GSP MSE loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        # C-CHAIN auxiliary step (two-step formulation).
        # After the MSE optimizer step, re-run the head on the same batch and
        # penalize the L2 distance from the pre-step snapshot. A second
        # backward+step is used so the C-CHAIN gradient does NOT mix with the
        # MSE gradient inside the same computation graph — this matches the
        # paper's "run optimizer on churn loss separately" interpretation and
        # avoids modifying the MSE loss value that gets logged.
        # Guard: recurrent path skipped (RDDPGActorNetwork forward signature
        # differs and its use case does not yet have plasticity concerns).
        if cchain_lambda > 0.0 and not recurrent:
            post_outputs = networks['actor'].forward(states)
            if post_outputs.dim() != pre_outputs.dim():
                # Normalize shape (batch,) → (batch, 1) to match pre_outputs
                post_outputs = post_outputs.unsqueeze(-1) if post_outputs.dim() == 1 else post_outputs
                pre_outputs = pre_outputs.unsqueeze(-1) if pre_outputs.dim() == 1 else pre_outputs
            cchain_loss = cchain_lambda * F.mse_loss(post_outputs, pre_outputs)
            networks['actor'].optimizer.zero_grad()
            cchain_loss.backward()
            _check_nan(cchain_loss, f"GSP C-CHAIN loss at step {networks['learn_step_counter']}")
            networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        # Phase 4 — loss-step correlation diagnostic.
        # Compute Pearson correlation between the FRESH forward-pass predictions
        # (the same preds that produced the MSE loss) and the replay-buffer labels.
        # This is intentionally different from gsp_pred_target_corr in hdf5_logger,
        # which accumulates actor-input-path predictions over a full episode (a
        # different code path with a 1-timestep lag). Computing per-batch here and
        # aggregating in the caller lets us compare "is the loss-path head actually
        # learning?" vs "is the actor-input path measurement broken?"
        #
        # Safety contract:
        # - Uses T.no_grad() / detach — zero gradient graph impact.
        # - NaN/zero-variance guard: returns float("nan") when undefined.
        # - Shape agnostic: flattens both arrays before corrcoef.
        # - Recurrent path: preds/labels_shaped not available in that scope,
        #   so we skip and return nan for consistency.
        batch_corr: float = float("nan")
        if not recurrent:
            with T.no_grad():
                _pred_np = preds.detach().cpu().numpy().flatten()
                _lbl_np = labels_shaped.detach().cpu().numpy().flatten()
                if _pred_np.size > 1:
                    _STD_TOL = 1e-12
                    _p_std = float(np.nanstd(_pred_np))
                    _l_std = float(np.nanstd(_lbl_np))
                    if _p_std > _STD_TOL and _l_std > _STD_TOL:
                        _mask = np.isfinite(_pred_np) & np.isfinite(_lbl_np)
                        if _mask.sum() > 1:
                            batch_corr = float(np.corrcoef(_pred_np[_mask], _lbl_np[_mask])[0, 1])

        return loss.item(), batch_corr

    def _update_jepa_target_encoder(self, tau: float) -> None:
        """EMA update: target_p ← tau * target_p + (1 - tau) * online_p.

        Args:
            tau: EMA decay coefficient (e.g. 0.995). Higher = slower update.
        """
        with T.no_grad():
            for online_p, target_p in zip(
                self.gsp_encoder_online.parameters(),
                self.gsp_encoder_target.parameters(),
            ):
                target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

    def learn_gsp_jepa(self, networks: dict):
        """Train the JEPA latent-space GSP head.

        Samples (state_t, state_{t+k}) pairs from the JEPA replay buffer
        (state_t in the 'state' slot, state_{t+k} in the 'action' slot by
        convention). Computes:

            z_t     = encoder_online(state_t)           # online encoding
            z_pred  = predictor(z_t)                    # predicted future latent
            z_target = encoder_target(state_{t+k}).detach()   # EMA target

            loss_pred = MSE(z_pred, z_target)           # latent prediction loss

        Optional VICReg variance + covariance penalties on z_t are added
        when self.gsp_vicreg_enabled is True (reusing existing helpers).

        After backward + optimizer step, the target encoder is updated via EMA.

        Returns:
            Tuple (loss_float, latent_stats_dict) where latent_stats_dict has:
                {var: float, rank: float, pred_mse: float}
        """
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return None

        vicreg_enabled = self.gsp_vicreg_enabled
        tau = self.gsp_encoder_ema_tau
        enc_device = self.gsp_encoder_online.device

        # Sample directly from the JEPA replay buffer rather than going through
        # sample_memory(), which requires a 'actor' or 'q_eval' key in networks
        # to determine device. JEPA networks dict has neither — device comes from
        # the encoder module itself.
        result = networks['replay'].sample_buffer(self.gsp_batch_size)
        raw_states, raw_future, _, _, _ = result[0], result[1], result[2], result[3], result[4]
        states = T.as_tensor(raw_states, dtype=T.float32).to(enc_device)
        # future_states: stored in the 'action' slot by convention (state_{t+k})
        future_states = T.as_tensor(raw_future, dtype=T.float32).to(enc_device)

        # Forward through online encoder + predictor
        z_t = self.gsp_encoder_online(states)
        z_pred = self.gsp_predictor(z_t)
        # SimNorm consistency: z_t/z_target already lie on the simplex (encoder
        # applies SimNorm in forward); project the predictor output onto the same
        # simplex so both operands of the self-prediction MSE match. No-op off.
        if self.gsp_jepa_simnorm:
            z_pred = simnorm(z_pred, self.gsp_jepa_simnorm_group_size)

        # Target: forward through frozen target encoder
        with T.no_grad():
            z_target = self.gsp_encoder_target(future_states)

        loss_pred = F.mse_loss(z_pred, z_target)
        loss = loss_pred

        # Optional VICReg on online encoder output z_t
        if vicreg_enabled:
            var_coef = self.gsp_vicreg_var_coef
            cov_coef = self.gsp_vicreg_cov_coef
            # target_std: 1.0 (standard VICReg default) — latent lives in
            # unbounded linear space so label-std normalization is not needed.
            var_loss = vicreg_variance_loss(z_t, target_std=1.0)
            cov_loss = vicreg_covariance_loss(z_t)
            loss = loss_pred + var_coef * var_loss + cov_coef * cov_loss

        self._jepa_online_optimizer.zero_grad()
        loss.backward()
        _check_nan(loss, f"JEPA loss at step {networks['learn_step_counter']}")
        self._jepa_online_optimizer.step()

        # EMA update of target encoder
        self._update_jepa_target_encoder(tau)

        networks['learn_step_counter'] += 1

        # Compute latent statistics (no grad)
        with T.no_grad():
            latent_var = float(z_t.var(dim=0).mean().item())
            # Approximate rank: number of singular values above 1% of max
            z_cpu = z_t.detach().cpu()
            try:
                sv = T.linalg.svdvals(z_cpu)
                rank = float((sv > sv[0] * 0.01).sum().item())
            except Exception:
                rank = float("nan")
            pred_mse = float(loss_pred.item())

        latent_stats = {
            'var': latent_var,
            'rank': rank,
            'pred_mse': pred_mse,
        }
        return loss.item(), latent_stats

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon-self.eps_dec, self.eps_min)

    def store_transition(self, s, a, r, s_, d, networks, gsp_obs=None, gsp_label=None, phi=None):
        # Only forward phi when set (SF path). The SequenceReplayBuffer / attention
        # buffers used by RDDPG/GSP do not accept a phi kwarg, so passing it
        # unconditionally would break every non-SF store. Keeps the legacy call
        # signature byte-identical when GSP_SF_ENABLED is off.
        if phi is None:
            networks['replay'].store_transition(s, a, r, s_, d, gsp_obs=gsp_obs, gsp_label=gsp_label)
        else:
            networks['replay'].store_transition(s, a, r, s_, d, gsp_obs=gsp_obs, gsp_label=gsp_label, phi=phi)
    
    def store_attention_transition(self, s, y, networks):
        networks['replay'].store_transition(s, y)

    def sample_memory(self, networks):
        result = networks['replay'].sample_buffer(self.batch_size)
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            device = networks['q_eval'].device
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG', 'TD3'}:
            device = networks['actor'].device

        if len(result) == 7:
            # Two sources of 7-value returns:
            # 1. SequenceReplayBuffer: items 5 and 6 are h_batch, c_batch — lists
            #    of tuples (hidden states), not numpy arrays.
            # 2. ReplayBuffer with gsp_obs_size > 0: items 5 and 6 are numpy
            #    arrays (gsp_obs, gsp_labels). learn_DDQN_e2e calls sample_buffer
            #    directly to get these — legacy callers only need the first 5.
            extra5 = result[5]
            if isinstance(extra5, np.ndarray):
                # E2E replay path: discard gsp_obs and gsp_labels for legacy callers.
                states, actions, rewards, states_, dones = (
                    result[0], result[1], result[2], result[3], result[4]
                )
                # as_tensor avoids an intermediate CPU copy for float32 arrays;
                # actions may be int-typed so pre-convert via numpy before as_tensor
                # to avoid a dtype-change + device-transfer race on accelerator backends.
                states = T.as_tensor(states, dtype=T.float32).to(device)
                actions = T.as_tensor(np.asarray(actions, dtype=np.float32)).to(device)
                rewards = T.as_tensor(rewards, dtype=T.float32).to(device)
                states_ = T.as_tensor(states_, dtype=T.float32).to(device)
                dones = T.as_tensor(dones).to(device)
                return states, actions, rewards, states_, dones
            else:
                # Sequence replay path: return all 7 (h_batch, c_batch are tensors/tuples).
                states, actions, rewards, states_, dones, h_batch, c_batch = result
                states = T.as_tensor(states, dtype=T.float32).to(device)
                actions = T.as_tensor(np.asarray(actions, dtype=np.float32)).to(device)
                rewards = T.as_tensor(rewards, dtype=T.float32).to(device)
                states_ = T.as_tensor(states_, dtype=T.float32).to(device)
                dones = T.as_tensor(dones).to(device)
                return states, actions, rewards, states_, dones, h_batch, c_batch
        else:
            states, actions, rewards, states_, dones = result
            states = T.as_tensor(states, dtype=T.float32).to(device)
            actions = T.as_tensor(np.asarray(actions, dtype=np.float32)).to(device)
            rewards = T.as_tensor(rewards, dtype=T.float32).to(device)
            states_ = T.as_tensor(states_, dtype=T.float32).to(device)
            dones = T.as_tensor(dones).to(device)
            return states, actions, rewards, states_, dones

    def sample_attention_memory(self, networks):
        observations, labels = networks['replay'].sample_buffer(self.batch_size)
        observations = T.as_tensor(observations, dtype=T.float32).to(networks['attention'].device)
        labels = T.as_tensor(labels, dtype=T.float32).to(networks['attention'].device)
        return observations, labels