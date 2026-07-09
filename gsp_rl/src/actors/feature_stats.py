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

Numerics
--------
Per-dimension Welford online mean/variance so the running estimate is exact
(no EMA horizon to tune) and matches an offline mean/var of the same samples.
Before the first update (``count == 0``) ``standardize`` is the identity — there
are no stats yet, so acting during the initial warmup sees the raw feature rather
than a divide-by-nothing artifact.
"""

from __future__ import annotations

import numpy as np


class RunningStandardizer:
    """Per-dimension running standardizer (Welford mean/variance).

    Parameters
    ----------
    dim : int
        Feature width (K). One (mean, M2) pair is tracked per dimension.
    eps : float
        Variance floor added under the sqrt for numerical stability.
    """

    __slots__ = ("dim", "eps", "count", "_mean", "_m2")

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        if dim < 1:
            raise ValueError(f"RunningStandardizer dim must be >= 1, got {dim}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.count: int = 0
        self._mean = np.zeros(self.dim, dtype=np.float64)
        self._m2 = np.zeros(self.dim, dtype=np.float64)

    # --- read-only views of the current (frozen) stats ---
    @property
    def mean(self) -> np.ndarray:
        """Current running mean, shape (dim,). Zeros before any update."""
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """Current running (population) variance, shape (dim,).

        Zeros before any update and while count < 2 (a single sample has no
        variance); ``standardize`` then divides by sqrt(eps) only.
        """
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
        # Chan et al. parallel/batch Welford: combine the running aggregate with
        # the batch aggregate in one shot (numerically stable, order-independent).
        n_b = arr.shape[0]
        if n_b == 0:
            return
        mean_b = arr.mean(axis=0)
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
        if _is_torch(x):
            import torch as T

            mean_t = T.as_tensor(self._mean, dtype=x.dtype, device=x.device)
            std_t = T.as_tensor(
                np.sqrt(self.var + self.eps), dtype=x.dtype, device=x.device
            )
            return (x - mean_t) / std_t
        arr = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var + self.eps).astype(np.float32)
        out = (arr - self._mean.astype(np.float32)) / std
        return out.astype(np.float32, copy=False)


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _to_numpy(x) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
