"""Per-episode plasticity and representation diagnostics.

Pure functions that take a network + a fixed eval batch of states, return dicts of
Python floats. Designed to be called once per diagnostic episode and the result
dict dropped into HDF5Logger as per-episode attrs.

See docs/specs/2026-04-17-diagnostics-instrumentation.md for the spec.

Metrics
-------
- ``compute_fau``               dormant-neuron fraction per ReLU layer (Sokar 2023)
- ``compute_overactive_fau``    over-active fraction per ReLU layer (Qin 2024 MARL signal)
- ``compute_weight_norms``      L2 (Frobenius) norm per layer — Lyle 2024 regenerative criterion
- ``compute_effective_rank``    99% singular-value threshold on penultimate activations
- ``compute_q_action_gap``      Q(a*) − Q(a_next) over eval batch — Weng-Lee 2026 signal
- ``compute_gsp_pred_diversity``Shannon entropy of binned predictions — catches collapse-to-constant
- ``compute_hidden_norm``       Frobenius norm of LSTM hidden state over eval batch
- ``compute_attention_entropy`` Shannon entropy over attention weights (stub when unhookable)
- ``compute_grad_zero_fraction``fraction of weight gradients near zero per layer (He 2603.21173 OCP Thm 1)
- ``compute_churn``             L2 distance between activations under before/after state_dicts (Tang 2506.00592)
- ``compute_kfac_hessian_erank``effective rank of KFAC Gauss-Newton block-Hessian per layer (He 2509.22335)

Design notes
------------
- All forward passes run under ``torch.no_grad()`` — diagnostics never perturb training.
- Layer name resolution supports dotted paths (e.g., ``'actor.fc1'``, ``'ee.weight_ih_l0'``)
  for composite networks like RDDPGActorNetwork. ``_resolve_layer`` handles the traversal.
- For ``compute_effective_rank``, callers specify a method name (``penultimate_fn``)
  that returns the penultimate-layer activations. Production networks should expose
  such a method or the compute call should use a forward-hook alternative.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Internal helper: resolve dotted attribute paths on nn.Module trees
# --------------------------------------------------------------------------------------

def _resolve_layer(net: nn.Module, name: str) -> nn.Module | None:
    """Traverse ``net`` following dotted ``name`` (e.g. ``'actor.fc1'``).

    Returns the resolved submodule/attribute, or ``None`` if any segment of the
    path does not exist.
    """
    obj = net
    for part in name.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _safe_key(name: str) -> str:
    """Convert a dotted layer name to an underscore-joined metric key.

    ``'actor.fc1'`` -> ``'actor_fc1'``, ``'fc1'`` -> ``'fc1'``.
    Keeps metric key names valid as HDF5 attribute names and Python identifiers.
    """
    return name.replace('.', '_')


# --------------------------------------------------------------------------------------
# Dormant-neuron fraction (FAU = Fraction of Active Units — but we report the DORMANT
# fraction, i.e., 1 - FAU_active, to match the Sokar 2023 convention)
# --------------------------------------------------------------------------------------

def compute_fau(
    net: nn.Module,
    eval_batch: torch.Tensor,
    layer_names: list[str],
    tau_dead: float = 0.1,
) -> dict[str, float]:
    """Per-layer fraction of units whose post-ReLU activation is below ``tau_dead``
    for EVERY sample in the eval batch (Sokar 2023 definition).

    A unit is "dormant" iff its activation ≤ tau_dead across all samples. Since ReLU
    clamps the post-ReLU activation to [0, ∞), tau_dead=0.1 catches units that are
    never meaningfully firing.

    Args:
        net: network with named ``nn.Linear`` submodules matching ``layer_names``.
            The forward pass must apply ReLU immediately after each named layer;
            the canonical DQN/DDQN/DDPG trunks we use do exactly this.
            Layer names may use dotted paths (e.g., ``'actor.fc1'``) for composite
            networks like RDDPGActorNetwork.
        eval_batch: ``(N, input_dim)`` tensor of states to measure against.
        layer_names: ordered list of linear layer attribute names to probe.
        tau_dead: activation threshold below which a unit counts as dead. 0.1 per
            Sokar 2023's τ.

    Returns:
        ``{"fau_<layer_key>": float in [0, 1]}`` for each layer, where layer_key
        replaces dots with underscores (``'actor.fc1'`` -> ``'fau_actor_fc1'``).
    """
    net.eval()
    device = eval_batch.device
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(key: str):
        def hook(_module, _inp, out):
            # Capture post-Linear output; the network's own forward applies ReLU after.
            # We apply ReLU here ourselves to inspect the post-activation values.
            activations[key] = torch.relu(out).detach()
        return hook

    for name in layer_names:
        layer = _resolve_layer(net, name)
        if layer is None:
            continue
        key = _safe_key(name)
        hooks.append(layer.register_forward_hook(_make_hook(key)))

    try:
        with torch.no_grad():
            _ = net(eval_batch.to(device))
    finally:
        for h in hooks:
            h.remove()

    result: dict[str, float] = {}
    for name in layer_names:
        key = _safe_key(name)
        metric_key = f"fau_{key}"
        if key not in activations:
            result[metric_key] = float("nan")
            continue
        act = activations[key]  # shape (N, layer_dim)
        # A unit is dormant if its activation ≤ tau_dead for ALL samples.
        per_sample_dead = (act <= tau_dead)  # (N, D)
        unit_always_dead = per_sample_dead.all(dim=0)  # (D,)
        result[metric_key] = float(unit_always_dead.float().mean().item())
    return result


def compute_overactive_fau(
    net: nn.Module,
    eval_batch: torch.Tensor,
    layer_names: list[str],
    tau_over: float = 0.9,
) -> dict[str, float]:
    """Per-layer fraction of units whose post-ReLU activation is above
    ``tau_over * max_activation_in_layer`` for EVERY sample in the batch.

    This is the Qin 2024 MARL-specific signal: shared-policy networks often develop
    a small number of "dominant" units that always saturate, suppressing learning
    in the rest of the layer. Naive ReDo resets only dormant units and misses this.

    Args:
        tau_over: fraction of per-layer max activation above which a unit is
            considered over-active.
        layer_names: may use dotted paths (e.g., ``'actor.fc1'``); keys in the
            returned dict replace dots with underscores.
    """
    net.eval()
    device = eval_batch.device
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(key: str):
        def hook(_module, _inp, out):
            activations[key] = torch.relu(out).detach()
        return hook

    for name in layer_names:
        layer = _resolve_layer(net, name)
        if layer is None:
            continue
        key = _safe_key(name)
        hooks.append(layer.register_forward_hook(_make_hook(key)))

    try:
        with torch.no_grad():
            _ = net(eval_batch.to(device))
    finally:
        for h in hooks:
            h.remove()

    result: dict[str, float] = {}
    for name in layer_names:
        key = _safe_key(name)
        metric_key = f"overactive_{key}"
        if key not in activations:
            result[metric_key] = float("nan")
            continue
        act = activations[key]  # (N, D)
        # Per-layer max across all samples and units
        layer_max = act.max().item()
        if layer_max <= 1e-8:
            # Layer completely dead — no over-active units by definition
            result[metric_key] = 0.0
            continue
        threshold = tau_over * layer_max
        per_sample_over = (act >= threshold)  # (N, D)
        unit_always_over = per_sample_over.all(dim=0)  # (D,)
        result[metric_key] = float(unit_always_over.float().mean().item())
    return result


# --------------------------------------------------------------------------------------
# Weight norms (L2 / Frobenius per layer)
# --------------------------------------------------------------------------------------

def compute_weight_norms(
    net: nn.Module,
    layer_names: list[str],
) -> dict[str, float]:
    """Frobenius norm ‖W‖ for each named layer's weight matrix.

    Lyle 2024's "regenerative weight-norm criterion": plasticity interventions that
    produce sub-linear weight-norm growth across training transfer across tasks;
    those that don't, don't. Logging ‖W‖ per layer per episode is the data needed
    to evaluate that criterion on any training run.

    Args:
        layer_names: may use dotted paths (e.g., ``'actor.fc1'``, ``'ee.weight_ih_l0'``).
            For LSTM weight matrices (which are plain tensors, not nn.Module), the
            path resolves to the tensor directly via ``_resolve_layer``. Keys in the
            returned dict replace dots with underscores.
    """
    result: dict[str, float] = {}
    for name in layer_names:
        key = _safe_key(name)
        metric_key = f"wnorm_{key}"
        try:
            obj = _resolve_layer(net, name)
            if obj is None:
                result[metric_key] = float("nan")
                continue
            # obj may be an nn.Module with a .weight attr, or a raw tensor
            # (e.g., LSTM weight matrices accessed as 'ee.weight_ih_l0').
            if isinstance(obj, torch.Tensor):
                w = obj.detach()
            else:
                w = obj.weight.detach()
            result[metric_key] = float(torch.linalg.norm(w).item())
        except AttributeError:
            result[metric_key] = float("nan")
    return result


# --------------------------------------------------------------------------------------
# Effective rank of penultimate activations (99% SVD threshold)
# --------------------------------------------------------------------------------------

def compute_effective_rank(
    net: nn.Module,
    eval_batch: torch.Tensor,
    penultimate_fn: str = "penultimate",
    threshold: float = 0.99,
) -> float:
    """Number of singular values of the penultimate-activation matrix required to
    cover ``threshold`` fraction of the total variance.

    Representation rank is the empirical handle on Lyle 2023's "feature rank
    collapse is the predominant failure mode of RL trunks" finding. Kumar 2020
    uses this exact metric.

    Args:
        net: must expose a method named ``penultimate_fn`` (default ``"penultimate"``)
            that returns the post-ReLU activations of the layer immediately before
            the output projection.
        eval_batch: ``(N, input_dim)`` states.
        threshold: cumulative-variance cutoff (0.99 per Kumar 2020).

    Returns:
        Effective rank as an integer in ``[1, D]`` where D is penultimate dim.
        Returned as float for h5 attr compatibility.
    """
    net.eval()
    fn = getattr(net, penultimate_fn)
    with torch.no_grad():
        acts = fn(eval_batch)  # (N, D)
    acts_np = acts.detach().cpu().numpy().astype(np.float64)

    # Center the activations per-feature — standard pre-SVD step for rank estimation
    acts_centered = acts_np - acts_np.mean(axis=0, keepdims=True)

    # If the batch is entirely constant or zero, rank is 0 — cover it explicitly
    if np.allclose(acts_centered, 0.0, atol=1e-10):
        return 0.0

    # Singular-value decomposition. np.linalg.svd returns singular values in descending order.
    try:
        _, s, _ = np.linalg.svd(acts_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")

    # Cumulative variance explained = cumsum(s^2) / sum(s^2)
    s_squared = s ** 2
    total = s_squared.sum()
    if total <= 1e-20:
        return 0.0
    cumvar = np.cumsum(s_squared) / total
    # Number of singular values needed to reach threshold
    rank = int(np.searchsorted(cumvar, threshold)) + 1
    return float(rank)


# --------------------------------------------------------------------------------------
# Q-value action gap (Weng-Lee 2026 cooperation-collapse signal)
# --------------------------------------------------------------------------------------

def compute_q_action_gap(
    q_network: nn.Module,
    eval_batch: torch.Tensor,
) -> dict[str, float]:
    """Q-value gap between the best and second-best action, averaged over the batch.

    Weng & Lee 2026 "Cooperation Collapse in Shared-Policy Multi-Agent DQN" identifies
    ``Q(a*) − Q(a_next_best) → 0`` as the representational-entanglement signature that
    is ORTHOGONAL to plasticity loss (dormant-neuron fraction). Dropping to 0 while
    FAU stays healthy means the network hasn't lost capacity — it's lost
    decidability between behaviors.

    Returns:
        ``q_action_gap_mean``, ``q_action_gap_std``, ``q_max_mean``
    """
    q_network.eval()
    with torch.no_grad():
        q = q_network(eval_batch)  # (N, A)
    # Sort per-sample descending, take top two
    sorted_q, _ = torch.sort(q, dim=1, descending=True)
    best = sorted_q[:, 0]
    second = sorted_q[:, 1] if sorted_q.shape[1] >= 2 else torch.zeros_like(best)
    gaps = best - second

    return {
        "q_action_gap_mean": float(gaps.mean().item()),
        "q_action_gap_std": float(gaps.std(unbiased=False).item()),
        "q_max_mean": float(best.mean().item()),
    }


# --------------------------------------------------------------------------------------
# GSP prediction diversity (Shannon entropy of binned predictions)
# --------------------------------------------------------------------------------------

def compute_gsp_pred_diversity(
    predictions: np.ndarray,
    n_bins: int = 10,
    low: float = -1.0,
    high: float = 1.0,
) -> float:
    """Shannon entropy (natural log) of per-timestep GSP predictions binned into
    ``n_bins`` equal-width bins over ``[low, high]``.

    A constant prediction → entropy = 0; uniformly varied predictions → log(n_bins).
    Gives a scalar per-episode summary that distinguishes ``out_std=0.07 but all
    the same value`` (collapse) from ``out_std=0.07 but actually varying``.
    """
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if preds.size == 0:
        return 0.0
    # Bin into [low, high]; out-of-range values get clipped to the edges.
    preds_clipped = np.clip(preds, low, high)
    counts, _ = np.histogram(preds_clipped, bins=n_bins, range=(low, high))
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Shannon entropy with natural log; 0 * log(0) = 0 by convention.
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -float((probs * logs).sum())
    return entropy


# --------------------------------------------------------------------------------------
# LSTM hidden-state norm (for EnvironmentEncoder / RDDPG networks)
# --------------------------------------------------------------------------------------

def compute_hidden_norm(
    net: nn.Module,
    eval_batch: torch.Tensor,
) -> float:
    """Frobenius norm of the mean LSTM hidden state across the eval batch.

    Passes ``eval_batch`` through ``net`` treating each row as a single-step
    observation sequence (seq_len=1). The LSTM returns ``(h_n, c_n)``; we use
    ``h_n`` (the hidden state, not the cell state) and compute the mean across
    the batch dimension before taking the Frobenius norm.

    This gives a scalar summary of how saturated the LSTM hidden state is — a
    proxy for capacity utilization distinct from the FAU metric (which targets
    ReLU units, not LSTM gates).

    Args:
        net: Any ``nn.Module`` whose forward returns ``(output, (h_n, c_n))``
            when called with a ``(batch, seq_len, input_size)`` tensor.
            Typically ``EnvironmentEncoder``.
        eval_batch: ``(N, input_dim)`` tensor of state observations.

    Returns:
        Frobenius norm of ``mean_h_n`` as a Python float.
        Returns ``float('nan')`` if the network's forward does not return an
        ``(output, (h_n, c_n))`` tuple.
    """
    net.eval()
    device = eval_batch.device
    # Wrap each sample as a single-step sequence: (N, 1, input_dim)
    x = eval_batch.unsqueeze(1).to(device)
    try:
        with torch.no_grad():
            out = net(x)
        if not (isinstance(out, tuple) and len(out) == 2):
            return float("nan")
        _, hidden = out
        if not (isinstance(hidden, tuple) and len(hidden) == 2):
            return float("nan")
        h_n, _ = hidden  # h_n shape: (num_layers, batch, hidden_size)
        # Use the top layer's hidden state (last layer index)
        h_top = h_n[-1]  # (batch, hidden_size)
        mean_h = h_top.mean(dim=0)  # (hidden_size,)
        return float(torch.linalg.norm(mean_h).item())
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------------------
# Attention entropy (for AttentionEncoder / A-GSP networks)
# --------------------------------------------------------------------------------------

def compute_attention_entropy(
    net: nn.Module,
    eval_batch: torch.Tensor,
) -> float:
    """Shannon entropy over attention weights, averaged across heads and samples.

    Captures whether the attention distribution is uniform (high entropy, attending
    broadly across the sequence) or peaked (low entropy, attending to a single
    position). Entropy collapse → the encoder has stopped using temporal context.

    Implementation: registers a forward hook on ``net.layers[0].attention.softmax``
    (the SelfAttention softmax inside the first TransformerBlock) to capture the
    attention weight tensor. If the hook target cannot be resolved (wrong
    architecture), returns ``float('nan')`` — documented as a TODO.

    Args:
        net: ``AttentionEncoder`` (or any module with a compatible attention path).
        eval_batch: ``(N, seq_len, obs_dim)`` tensor of observation sequences.
            If ``eval_batch`` is 2-D ``(N, obs_dim)``, it is treated as a single
            observation per sample (seq_len=1).

    Returns:
        Mean Shannon entropy of the attention distribution as a Python float.
        Returns ``float('nan')`` if the attention weights cannot be captured.

    TODO: The hook target path (``layers[0].attention.softmax``) is
    AttentionEncoder-specific. Generalizing to arbitrary architectures requires
    a registered hook interface on the module class — planned as a followup.
    """
    net.eval()
    device = eval_batch.device

    # Resolve the softmax module inside the first TransformerBlock's SelfAttention
    try:
        softmax_mod = net.layers[0].attention.softmax
    except (AttributeError, IndexError):
        return float("nan")

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        # out shape: (N, heads, query_len, key_len)
        captured['attn'] = out.detach()

    handle = softmax_mod.register_forward_hook(_hook)

    # Ensure input is 3-D (N, seq_len, obs_dim)
    if eval_batch.dim() == 2:
        x = eval_batch.unsqueeze(1).to(device)
    else:
        x = eval_batch.to(device)

    try:
        with torch.no_grad():
            _ = net(x)
    except Exception:
        return float("nan")
    finally:
        handle.remove()

    if 'attn' not in captured:
        return float("nan")

    attn = captured['attn']  # (N, heads, query_len, key_len)
    # Compute Shannon entropy over key_len dimension for each (sample, head, query)
    # Then average across all of those dimensions.
    # Clamp for numerical safety (softmax should already be ≥ 0, but fp edge cases)
    p = attn.clamp(min=1e-12)
    # Shannon entropy: H = -sum(p * log(p)) along last dim
    entropy_per_qhk = -(p * p.log()).sum(dim=-1)  # (N, heads, query_len)
    return float(entropy_per_qhk.mean().item())


# --------------------------------------------------------------------------------------
# Gradient zero fraction per layer (He 2603.21173 OCP Theorem 1)
# --------------------------------------------------------------------------------------

def compute_grad_zero_fraction(
    net: nn.Module,
    loss_fn,
    eval_batch: torch.Tensor,
    layer_names: list[str],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Per-layer fraction of weight gradient entries with |grad| < eps.

    Theory: OCP Theorem 1 (He 2603.21173) proves dormancy ≡ zero-gradient
    equivalence under continuity assumptions. This metric should track FAU
    closely; divergence between the two signals indicates a measurement artifact
    or that the network has lost gradient flow without losing unit activation.

    Implementation:
    - Zeroes gradients, runs a forward pass through ``net`` with ``eval_batch``.
    - Calls ``loss_fn(output, target)`` where target is zeros (we want to
      detect gradient flow shut-off, not predictive accuracy).
    - Calls ``.backward()`` to populate ``.grad`` on each parameter.
    - For each named linear layer, counts the fraction of weight gradient
      entries whose absolute value is below ``eps``.
    - Restores zero-gradients after so this never perturbs training.

    Args:
        net: ``nn.Module`` with named ``nn.Linear`` submodules.
        loss_fn: callable(output, target) -> scalar Tensor. A reasonable default
            is ``torch.nn.functional.mse_loss`` with a zero target.
        eval_batch: ``(N, input_dim)`` float tensor of states.
        layer_names: list of layer names to probe (may use dotted paths).
        eps: threshold below which a gradient entry counts as "zero".

    Returns:
        ``{"grad_zero_<layer_key>": float in [0, 1]}`` for each layer, where
        layer_key replaces dots with underscores.
    """
    device = eval_batch.device
    net.train()  # Need grad-mode; zero any existing grads first
    net.zero_grad()

    try:
        output = net(eval_batch.to(device))
        # Use zero target — we probe gradient flow, not prediction quality.
        if isinstance(output, tuple):
            # Some networks return (out, hidden); use the first element.
            output = output[0]
        target = torch.zeros_like(output)
        loss = loss_fn(output, target)
        loss.backward()
    except Exception:
        net.zero_grad()
        net.eval()
        return {f"grad_zero_{_safe_key(n)}": float("nan") for n in layer_names}

    result: dict[str, float] = {}
    for name in layer_names:
        key = _safe_key(name)
        metric_key = f"grad_zero_{key}"
        layer = _resolve_layer(net, name)
        if layer is None or not hasattr(layer, 'weight') or layer.weight.grad is None:
            result[metric_key] = float("nan")
            continue
        grad = layer.weight.grad.detach()
        result[metric_key] = float((grad.abs() < eps).float().mean().item())

    net.zero_grad()
    net.eval()
    return result


# --------------------------------------------------------------------------------------
# Activation churn: L2 distance of activations under before vs after state_dicts
# (Tang 2506.00592 C-CHAIN)
# --------------------------------------------------------------------------------------

def compute_churn(
    net: nn.Module,
    eval_batch: torch.Tensor,
    before_state_dict: dict,
    after_state_dict: dict,
    layer_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """L2 distance between activations on ``eval_batch`` under two state dicts.

    Theory: C-CHAIN (Tang 2506.00592, ICML 2025) — high churn predicts
    plasticity loss in the next ~100k training steps. High churn means the
    network's functional representation changes significantly per update step,
    which exhausts capacity faster.

    Caller is responsible for snapshotting before/after state_dicts via
    ``copy.deepcopy(net.state_dict())`` around a training step.

    Args:
        net: the network to evaluate on.
        eval_batch: ``(N, input_dim)`` state batch.
        before_state_dict: state dict captured before the training step.
        after_state_dict: state dict captured after the training step.
        layer_names: optional list of layer names (dotted) to measure per-layer
            churn via forward hooks. If ``None``, only final-output churn is
            returned.

    Returns:
        ``{"churn_output": float}`` always present. If ``layer_names`` given,
        also ``{"churn_<layer_key>": float}`` per layer.
        Frobenius norm of activation difference divided by sqrt(N) for
        batch-size normalization.
    """
    device = eval_batch.device
    x = eval_batch.to(device)
    result: dict[str, float] = {}

    def _capture_output_and_layers(state_dict, lnames):
        """Load state_dict, run forward, capture output + per-layer activations."""
        # Load weights into net
        net.load_state_dict(state_dict)
        net.eval()
        activations: dict[str, torch.Tensor] = {}
        hooks = []

        if lnames:
            def _make_hook(key: str):
                def hook(_module, _inp, out):
                    act = out
                    if isinstance(act, tuple):
                        act = act[0]
                    activations[key] = act.detach()
                return hook

            for ln in lnames:
                layer = _resolve_layer(net, ln)
                if layer is not None:
                    k = _safe_key(ln)
                    hooks.append(layer.register_forward_hook(_make_hook(k)))

        try:
            with torch.no_grad():
                out = net(x)
            if isinstance(out, tuple):
                out = out[0]
            final_out = out.detach()
        finally:
            for h in hooks:
                h.remove()

        return final_out, activations

    try:
        out_before, acts_before = _capture_output_and_layers(before_state_dict, layer_names)
        out_after, acts_after = _capture_output_and_layers(after_state_dict, layer_names)
    except Exception:
        return {"churn_output": float("nan")}

    N = float(x.shape[0])
    diff_out = out_after - out_before
    result["churn_output"] = float(torch.linalg.norm(diff_out).item()) / (N ** 0.5)

    if layer_names:
        for ln in layer_names:
            k = _safe_key(ln)
            metric_key = f"churn_{k}"
            if k in acts_before and k in acts_after:
                diff = acts_after[k] - acts_before[k]
                result[metric_key] = float(torch.linalg.norm(diff).item()) / (N ** 0.5)
            else:
                result[metric_key] = float("nan")

    return result


# --------------------------------------------------------------------------------------
# KFAC Hessian effective rank per layer (He 2509.22335 Theorem 6.2)
# --------------------------------------------------------------------------------------

def compute_kfac_hessian_erank(
    net: nn.Module,
    eval_batch: torch.Tensor,
    layer_names: list[str],
    threshold: float = 0.99,
) -> dict[str, float]:
    """Per-layer effective rank of the Gauss-Newton block-Hessian via KFAC approximation.

    Theory: He et al. 2509.22335 Theorem 6.2 — ``rank(H) ≤ P − k_τ(I+O+1)``
    where k_τ is the dead-neuron count. Dead neurons cap the Hessian rank; this
    metric probes the same collapse from the loss-curvature perspective.

    KFAC approximation per layer:
        H_l ≈ A_l ⊗ G_l
    where:
        A_l = E[a_l a_l^T]    — input activation covariance (d_in x d_in)
        G_l = E[g_l g_l^T]    — pre-activation gradient covariance (d_out x d_out)

    The effective rank of H_l ≈ erank(A_l) · erank(G_l) by the Kronecker rank
    product property.

    Implementation:
    - Captures pre-activation inputs (a_l) via forward hooks.
    - Captures pre-activation gradients (g_l = ∂L/∂z_l) via backward hooks.
    - Computes A_l and G_l as outer-product averages over the batch.
    - Applies the same 99%-SVD effective rank computation as compute_effective_rank.
    - Zeroes gradients before and after so this never perturbs training.

    Args:
        net: network with named ``nn.Linear`` submodules.
        eval_batch: ``(N, input_dim)`` batch. Spec recommends N=256 to bound
            KFAC compute (the matrices are d_in x d_in and d_out x d_out, not N x N).
        layer_names: list of layer names to probe (dotted paths supported).
        threshold: cumulative-variance cutoff for effective rank (0.99 per spec).

    Returns:
        ``{"kfac_erank_<layer_key>": float}`` per layer — the product of
        erank(A_l) and erank(G_l). Also returns
        ``{"kfac_erank_A_<layer_key>": float}`` and
        ``{"kfac_erank_G_<layer_key>": float}`` for separate analysis.
    """
    device = eval_batch.device
    x = eval_batch.to(device)

    net.train()  # backward requires grad mode
    net.zero_grad()

    inputs_per_layer: dict[str, list[torch.Tensor]] = {}
    grads_per_layer: dict[str, list[torch.Tensor]] = {}
    hooks = []

    for name in layer_names:
        layer = _resolve_layer(net, name)
        if layer is None or not isinstance(layer, nn.Linear):
            continue
        key = _safe_key(name)
        inputs_per_layer[key] = []
        grads_per_layer[key] = []

        def _make_forward_hook(k: str):
            def hook(_module, inp, _out):
                # inp is a tuple; first element is the layer input (pre-weight matmul)
                a = inp[0].detach()  # (N, d_in)
                inputs_per_layer[k].append(a)
            return hook

        def _make_backward_hook(k: str):
            def hook(_module, _grad_in, grad_out):
                # grad_out[0] is the gradient w.r.t. the layer's output (pre-activation)
                g = grad_out[0].detach()  # (N, d_out)
                grads_per_layer[k].append(g)
            return hook

        hooks.append(layer.register_forward_hook(_make_forward_hook(key)))
        hooks.append(layer.register_full_backward_hook(_make_backward_hook(key)))

    try:
        output = net(x)
        if isinstance(output, tuple):
            output = output[0]
        target = torch.zeros_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
    except Exception:
        for h in hooks:
            h.remove()
        net.zero_grad()
        net.eval()
        return {f"kfac_erank_{_safe_key(n)}": float("nan") for n in layer_names}
    finally:
        for h in hooks:
            h.remove()

    net.zero_grad()
    net.eval()

    result: dict[str, float] = {}

    def _svd_erank(mat: np.ndarray, thr: float) -> float:
        """99%-SVD effective rank of a square matrix."""
        if np.allclose(mat, 0.0, atol=1e-10):
            return 0.0
        try:
            _, s, _ = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return float("nan")
        s2 = s ** 2
        total = s2.sum()
        if total <= 1e-20:
            return 0.0
        cumvar = np.cumsum(s2) / total
        return float(int(np.searchsorted(cumvar, thr)) + 1)

    for name in layer_names:
        key = _safe_key(name)
        metric_kfac = f"kfac_erank_{key}"
        metric_A = f"kfac_erank_A_{key}"
        metric_G = f"kfac_erank_G_{key}"

        if key not in inputs_per_layer or not inputs_per_layer[key]:
            result[metric_kfac] = float("nan")
            result[metric_A] = float("nan")
            result[metric_G] = float("nan")
            continue

        # Stack batch activations from possibly multiple accumulation calls
        A_list = inputs_per_layer[key]
        G_list = grads_per_layer.get(key, [])

        try:
            A_batch = torch.cat(A_list, dim=0).cpu().float()  # (N, d_in)
            N = A_batch.shape[0]
            # A_l = E[a a^T] = (1/N) * A^T A, giving (d_in, d_in)
            A_cov = (A_batch.T @ A_batch / N).numpy().astype(np.float64)
            erank_A = _svd_erank(A_cov, threshold)
        except Exception:
            result[metric_kfac] = float("nan")
            result[metric_A] = float("nan")
            result[metric_G] = float("nan")
            continue

        if G_list:
            try:
                G_batch = torch.cat(G_list, dim=0).cpu().float()  # (N, d_out)
                N_g = G_batch.shape[0]
                G_cov = (G_batch.T @ G_batch / N_g).numpy().astype(np.float64)
                erank_G = _svd_erank(G_cov, threshold)
            except Exception:
                erank_G = float("nan")
        else:
            erank_G = float("nan")

        result[metric_A] = erank_A
        result[metric_G] = erank_G
        if math.isfinite(erank_A) and math.isfinite(erank_G):
            result[metric_kfac] = erank_A * erank_G
        else:
            result[metric_kfac] = float("nan")

    return result
