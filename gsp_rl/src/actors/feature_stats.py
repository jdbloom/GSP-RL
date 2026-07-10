"""Running per-feature standardization for the spliced GSP prediction.

Opt-in behind ``GSP_E2E_NORMALIZE_FEATURE``. The GSP prediction that is spliced
into the actor's Q-input is a tiny-magnitude scalar (std ~0.024) sitting among
O(1) egocentric observations. ``ACTOR_USE_LAYER_NORM`` normalizes the whole
input vector, not per-feature, so the actor cannot learn to weight a feature that
small. This class standardizes the spliced feature to ~unit variance so it lands
on the same scale as the surrounding obs.

Consistency contract
--------------------
The SAME ``RunningStandardizer`` instance MUST be used at both splice points:

1. Acting / rollout (``agent.make_agent_state``): standardize the GSP slot right
   before it is concatenated into the actor input.
2. E2E learning (``learn_DDQN_e2e`` / ``learn_TD3_e2e``): standardize the spliced
   prediction before it is concatenated into the augmented Q-input.

If acting and learning used different stats the actor would see two different
input distributions for the same feature and the Bellman update would be
internally inconsistent. The seam that guarantees a single shared instance is the
Actor object itself: RL-CT's ``Agent`` subclasses GSP-RL's ``Actor``, and the
same per-robot instance owns both ``make_agent_state`` (acting) and
``learn_*_e2e`` (learning). The instance lives on ``self.gsp_feature_stats``.

Update policy
-------------
Statistics are updated ONLY during training, from the fresh raw prediction batch
inside the learn splice. Acting standardizes with the current (frozen) stats and
never updates. At eval no learn step runs, so the stats are automatically frozen.
This keeps train/eval symmetric: eval uses the stats learned during training.

Persistence contract
--------------------
The stats are part of the policy: an actor trained on the standardized feature
is calibrated to ~unit-std input, so an eval process that reconstructs the
standardizer cold (``count == 0`` → ``standardize`` is the identity) feeds the
policy the RAW tiny-scale feature — a silent ~40x input-scale mismatch that
nulls the coupling in every arm of an ablation. Measured incident (2026-07-10):
every fresh-process ablation eval of the force-causal-use campaign ran with the
identity standardize, voiding the paired-gap verdict. ``save``/``restore``
serialize the full state (both Welford and EMA fields) to ``.npz``;
``Actor.save_model``/``load_model`` call them alongside the network
checkpoints whenever ``gsp_feature_stats`` is present.

Numerics
--------
Default mode (``ema_halflife == 0``): per-dimension Welford online mean/variance
so the running estimate is exact (no EMA horizon to tune) and matches an offline
mean/var of the same samples. Before the first update (``count == 0``)
``standardize`` is the identity — there are no stats yet, so acting during the
initial warmup sees the raw feature rather than a divide-by-nothing artifact.

EMA mode (``ema_halflife > 0``, opt-in via ``GSP_E2E_NORMALIZE_EMA_HALFLIFE``):
all-history Welford never forgets. Measured failure (2026-07-09, live cells): at
high ``GSP_E2E_LAMBDA`` (40000) the head's early outputs are large/noisy and
permanently inflate the running std, so the post-norm feature std reached only
0.17-0.39 instead of ~1.0 — the standardizer silently re-shrank the feature it
exists to normalize. In EMA mode the stats are a bias-corrected exponential
moving average (Adam-style) over UPDATE calls (i.e. learn steps, not samples):
per update ``t`` with batch mean ``m_t`` and batch raw second moment ``s_t``,

    ema_mean_t = beta * ema_mean_{t-1} + (1 - beta) * m_t
    ema_sq_t   = beta * ema_sq_{t-1}   + (1 - beta) * s_t
    mean = ema_mean_t / (1 - beta^t)
    var  = max(ema_sq_t / (1 - beta^t) - mean^2, 0)

with ``beta = 0.5 ** (1 / ema_halflife)`` so a batch's weight halves every
``ema_halflife`` updates. The bias correction ``1 / (1 - beta^t)`` makes the
warmup exact rather than zero-shrunk: at ``t == 1`` the stats equal the first
batch's mean/var (identical to Welford), and for ``t << ema_halflife`` they
approximate a near-uniform average of the batches seen so far — no separate
Welford warmup phase is needed. The ``count == 0`` identity behavior is
unchanged. ``ema_halflife == 0`` (the default) leaves the legacy Welford path
byte-identical for existing runs.
"""

from __future__ import annotations

import numpy as np


class RunningStandardizer:
    """Per-dimension running standardizer (Welford or EMA mean/variance).

    Parameters
    ----------
    dim : int
        Feature width (K). One (mean, M2) pair is tracked per dimension.
    eps : float
        Variance floor added under the sqrt for numerical stability.
    ema_halflife : float
        Exponential-moving-average half-life in UPDATE counts (learn steps).
        0 (default) = legacy all-history Welford, byte-identical to prior runs.
        > 0 = bias-corrected EMA mean/variance whose per-batch weight halves
        every ``ema_halflife`` updates, so the stats track the RECENT feature
        distribution and forget an inflated early phase (see module docstring).
    """

    __slots__ = (
        "dim", "eps", "count", "_mean", "_m2",
        "ema_halflife", "_beta", "_updates", "_ema_mean", "_ema_sq",
    )

    def __init__(
        self, dim: int, eps: float = 1e-5, ema_halflife: float = 0.0
    ) -> None:
        if dim < 1:
            raise ValueError(f"RunningStandardizer dim must be >= 1, got {dim}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.count: int = 0
        self._mean = np.zeros(self.dim, dtype=np.float64)
        self._m2 = np.zeros(self.dim, dtype=np.float64)
        self.ema_halflife = float(ema_halflife)
        if self.ema_halflife < 0:
            raise ValueError(
                f"RunningStandardizer ema_halflife must be >= 0, "
                f"got {ema_halflife}"
            )
        # EMA state (inert when ema_halflife == 0 — the legacy Welford fields
        # above remain the sole source of truth in that mode).
        self._beta = (
            0.5 ** (1.0 / self.ema_halflife) if self.ema_halflife > 0 else 0.0
        )
        self._updates: int = 0
        self._ema_mean = np.zeros(self.dim, dtype=np.float64)
        self._ema_sq = np.zeros(self.dim, dtype=np.float64)

    # --- read-only views of the current (frozen) stats ---
    @property
    def mean(self) -> np.ndarray:
        """Current running mean, shape (dim,). Zeros before any update.

        EMA mode: bias-corrected EMA of the batch means (exact batch mean at
        the first update)."""
        if self.ema_halflife > 0:
            if self._updates == 0:
                return np.zeros(self.dim, dtype=np.float64)
            return self._ema_mean / (1.0 - self._beta ** self._updates)
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """Current running (population) variance, shape (dim,).

        Welford mode: zeros before any update and while count < 2 (a single
        sample has no variance); ``standardize`` then divides by sqrt(eps) only.
        EMA mode: ``E_ema[x^2] - E_ema[x]^2`` with bias-corrected moments,
        clipped at 0 against floating-point cancellation.
        """
        if self.ema_halflife > 0:
            if self._updates == 0:
                return np.zeros(self.dim, dtype=np.float64)
            corr = 1.0 - self._beta ** self._updates
            mean_hat = self._ema_mean / corr
            return np.maximum(self._ema_sq / corr - mean_hat ** 2, 0.0)
        if self.count < 2:
            return np.zeros(self.dim, dtype=np.float64)
        return self._m2 / self.count

    @property
    def std(self) -> np.ndarray:
        """Current running std with the eps floor, shape (dim,)."""
        return np.sqrt(self.var + self.eps)

    def update(self, batch) -> None:
        """Fold a batch of feature vectors into the running mean/variance.

        Parameters
        ----------
        batch : array-like or torch.Tensor
            Shape (N, dim) or (N,) (treated as (N, 1) when dim == 1). torch
            tensors are detached to numpy first — updating stats must never
            create autograd edges.

        Update-only; does not standardize or return anything. Call this exactly
        once per learn step, on the RAW (pre-standardization) prediction batch.
        """
        arr = _to_numpy(batch)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1) if self.dim == 1 else arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"RunningStandardizer.update expected last dim {self.dim}, "
                f"got array with shape {arr.shape}"
            )
        arr = arr.astype(np.float64, copy=False)
        n_b = arr.shape[0]
        if n_b == 0:
            return
        mean_b = arr.mean(axis=0)
        if self.ema_halflife > 0:
            # EMA mode: fold this update's batch mean and raw second moment into
            # the (bias-corrected) exponential moving averages. The half-life
            # clock ticks once per UPDATE call (learn step), independent of the
            # batch size; ``count`` still counts samples so the ``count == 0``
            # identity and the learn-splice count contract are unchanged.
            self._updates += 1
            w = 1.0 - self._beta
            self._ema_mean = self._beta * self._ema_mean + w * mean_b
            self._ema_sq = self._beta * self._ema_sq + w * (arr ** 2).mean(axis=0)
            self.count += n_b
            return
        # Chan et al. parallel/batch Welford: combine the running aggregate with
        # the batch aggregate in one shot (numerically stable, order-independent).
        m2_b = ((arr - mean_b) ** 2).sum(axis=0)
        n_a = self.count
        if n_a == 0:
            self._mean = mean_b
            self._m2 = m2_b
            self.count = n_b
            return
        delta = mean_b - self._mean
        tot = n_a + n_b
        self._mean = self._mean + delta * (n_b / tot)
        self._m2 = self._m2 + m2_b + (delta ** 2) * (n_a * n_b / tot)
        self.count = tot

    def save(self, path: str) -> None:
        """Serialize the full standardizer state (Welford + EMA fields) to
        ``path`` as a ``.npz``. Called from ``Actor.save_model`` next to the
        network checkpoints — the stats are part of the policy (see the
        persistence contract in the module docstring)."""
        np.savez(
            path,
            dim=np.int64(self.dim),
            eps=np.float64(self.eps),
            count=np.int64(self.count),
            mean=self._mean,
            m2=self._m2,
            ema_halflife=np.float64(self.ema_halflife),
            updates=np.int64(self._updates),
            ema_mean=self._ema_mean,
            ema_sq=self._ema_sq,
        )

    def restore(self, path: str) -> None:
        """Load state saved by ``save`` into this instance, in place.

        Raises ``ValueError`` on a feature-width mismatch (restoring stats for
        a different K would silently mis-scale every dimension). ``eps`` and
        ``ema_halflife`` are restored from the file: the saved run's numerics
        define how the policy was trained, and the restoring process must
        reproduce them exactly.
        """
        with np.load(path) as z:
            file_dim = int(z["dim"])
            if file_dim != self.dim:
                raise ValueError(
                    f"RunningStandardizer.restore: file dim {file_dim} != "
                    f"instance dim {self.dim}"
                )
            self.eps = float(z["eps"])
            self.count = int(z["count"])
            self._mean = z["mean"].astype(np.float64)
            self._m2 = z["m2"].astype(np.float64)
            self.ema_halflife = float(z["ema_halflife"])
            self._beta = (
                0.5 ** (1.0 / self.ema_halflife)
                if self.ema_halflife > 0
                else 0.0
            )
            self._updates = int(z["updates"])
            self._ema_mean = z["ema_mean"].astype(np.float64)
            self._ema_sq = z["ema_sq"].astype(np.float64)

    def standardize(self, x):
        """Return ``(x - mean) / std`` using the current (frozen) stats.

        Does NOT update the stats. Preserves the input type: a torch tensor in →
        torch tensor out (grad-preserving; mean/std enter as constants), a numpy
        / python array in → numpy array out (used by the acting path).

        Before the first update (count == 0) this is the identity: the same
        object/value is returned unchanged, so a flag-on run whose stats have not
        warmed up yet still behaves sanely (and the acting path stays cheap).
        """
        if self.count == 0:
            return x
        # self.mean / self.var dispatch on the mode (Welford vs bias-corrected
        # EMA); in Welford mode they are the exact same values as before.
        mean = self.mean
        if _is_torch(x):
            import torch as T

            mean_t = T.as_tensor(mean, dtype=x.dtype, device=x.device)
            std_t = T.as_tensor(
                np.sqrt(self.var + self.eps), dtype=x.dtype, device=x.device
            )
            return (x - mean_t) / std_t
        arr = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var + self.eps).astype(np.float32)
        out = (arr - mean.astype(np.float32)) / std
        return out.astype(np.float32, copy=False)


def actor_gsp_feature_weight_diag(weight, n_obs: int, k: int) -> dict:
    """Measure how strongly the actor's first linear layer weights the spliced
    GSP prediction feature relative to a typical observation dimension.

    This is the headline causal-usage diagnostic for the E2E GSP investigation:
    the GSP prediction is well-learned but the causal ablation is null (the actor
    appears to ignore it). If the actor is genuinely relying on the spliced
    feature, the L2 norm of the first-layer weight COLUMNS feeding that feature
    should grow (and its per-dim ratio to the obs columns move away from ~0). If
    the actor ignores it, those columns stay near their init magnitude and the
    ratio stays flat/small.

    Layout contract
    ---------------
    The actor input is ``[obs (n_obs) | gsp (k)]`` so the first-layer weight
    ``W`` has shape ``(hidden, n_obs + k)``. The GSP prediction feeds the LAST
    ``k`` columns ``W[:, n_obs:]``; the obs feeds ``W[:, :n_obs]``.

    Parameters
    ----------
    weight : torch.Tensor or array-like
        First linear layer weight, shape ``(hidden, n_obs + k)`` (e.g. the DDQN
        Q-net ``fc1.weight`` or the TD3 actor ``fc1.weight``). Detached
        internally; no autograd edge is created.
    n_obs : int
        Number of raw observation input dims (columns ``[0, n_obs)``).
    k : int
        Width of the spliced GSP feature (columns ``[n_obs, n_obs + k)``).

    Returns
    -------
    dict with keys:
        ``actor_gsp_feature_weight_norm`` : float
            Frobenius norm of the GSP columns ``||W[:, n_obs:]||_F``.
        ``actor_obs_weight_norm_mean`` : float
            Mean over the obs columns of the per-column L2 norm
            ``mean_j ||W[:, j]||_2`` for ``j < n_obs``. A per-dim baseline.
        ``actor_gsp_weight_ratio`` : float
            Per-dim GSP-vs-obs reliance:
            ``mean_gsp_column_norm / actor_obs_weight_norm_mean`` where
            ``mean_gsp_column_norm = mean_j ||W[:, j]||_2`` for
            ``n_obs <= j < n_obs + k``. This puts the GSP columns and the obs
            columns on the SAME per-dimension footing regardless of ``k`` (for
            ``k == 1`` it equals ``gsp_col_norm / obs_col_norm``). ``~0`` → the
            actor ignores the prediction; growing → it relies on it.

    Notes
    -----
    Pure read of the weight values — no gradient, no in-place mutation. Callable
    under ``torch.no_grad()`` at the call site; also detaches defensively so a
    graph-carrying weight cannot leak an autograd edge into the diagnostics.
    """
    n_obs = int(n_obs)
    k = int(k)
    if n_obs < 0 or k < 1:
        raise ValueError(f"actor_gsp_feature_weight_diag needs n_obs>=0, k>=1; got n_obs={n_obs}, k={k}")
    if _is_torch(weight):
        import torch as T

        with T.no_grad():
            w = weight.detach()
            if w.dim() != 2 or w.shape[1] != n_obs + k:
                raise ValueError(
                    f"weight shape {tuple(w.shape)} incompatible with "
                    f"n_obs+k={n_obs + k} (expected (hidden, {n_obs + k}))"
                )
            col_norms = T.linalg.vector_norm(w, dim=0)  # per-column L2, shape (n_obs+k,)
            gsp_cols = col_norms[n_obs:]
            obs_cols = col_norms[:n_obs]
            gsp_feature_weight_norm = float(T.linalg.vector_norm(gsp_cols).item())
            obs_weight_norm_mean = (
                float(obs_cols.mean().item()) if n_obs > 0 else 0.0
            )
            mean_gsp_col_norm = float(gsp_cols.mean().item())
    else:
        w = np.asarray(weight, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != n_obs + k:
            raise ValueError(
                f"weight shape {tuple(w.shape)} incompatible with "
                f"n_obs+k={n_obs + k} (expected (hidden, {n_obs + k}))"
            )
        col_norms = np.linalg.norm(w, axis=0)
        gsp_cols = col_norms[n_obs:]
        obs_cols = col_norms[:n_obs]
        gsp_feature_weight_norm = float(np.linalg.norm(gsp_cols))
        obs_weight_norm_mean = float(obs_cols.mean()) if n_obs > 0 else 0.0
        mean_gsp_col_norm = float(gsp_cols.mean())
    ratio = (
        mean_gsp_col_norm / obs_weight_norm_mean
        if obs_weight_norm_mean > 0.0
        else 0.0
    )
    return {
        "actor_gsp_feature_weight_norm": gsp_feature_weight_norm,
        "actor_obs_weight_norm_mean": obs_weight_norm_mean,
        "actor_gsp_weight_ratio": ratio,
    }


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _to_numpy(x) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
