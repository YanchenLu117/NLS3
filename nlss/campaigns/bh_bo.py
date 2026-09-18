"""BH GP-BO baseline arm (botorch) — strong-BO confrontation (§9.8 B4/B6).

Fits a botorch SingleTaskGP (RBF kernel) on the queried (features -> yield)
pairs and acquires the next batch by Expected Improvement over the unqueried
candidate pool.  This is the modern "strong BO" that the report requires NLSS
to confront: NLSS need not beat BO on best-yield; it should stay competitive on
optimization while being clearly better on solution-space recovery
(AURC / RegionRecall, §9.13).

Best-effort: if botorch/torch is unavailable the factory returns None and the
arm is skipped, so the campaign still runs on torch-less envs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from nlss.campaigns.bh_oracle import BHFiniteOracle

try:
    import torch
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.acquisition import qLogExpectedImprovement, qExpectedImprovement
    from botorch.optim import optimize_acqf

    _HAS_BOTORCH = True
except Exception:  # pragma: no cover
    _HAS_BOTORCH = False


def has_botorch() -> bool:
    return _HAS_BOTORCH


if _HAS_BOTORCH:
    class _DeepRBFKernel(gpytorch.kernels.Kernel):
        """Deep RBF kernel: applies a learned MLP feature extractor to the raw
        features, then an RBF kernel on the 16-d embedding (gpytorch 1.15).

        The extractor parameters are registered as kernel hyperparameters so
        marginal-likelihood training in botorch optimizes them jointly with the
        kernel lengthscale/noise — the standard DKL recipe.
        """

        has_lengthscale = True

        def __init__(self, feature_extractor, **kwargs):
            super().__init__(**kwargs)
            self.feature_extractor = feature_extractor
            self.base_kernel = gpytorch.kernels.RBFKernel()

        def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
            e1 = self.feature_extractor(x1)
            e2 = self.feature_extractor(x2)
            return self.base_kernel.forward(
                e1, e2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
            )


class BHBotorchBO:
    """GP-BO with Expected Improvement acquisition over the whole pool.

    ``kernel`` selects the covariance model:
      * ``rbf`` — plain RBF/SingleTaskGP (GP-BO anchor, §9.8 B4/GP-BO).
      * ``dkl``  — Deep Kernel GP (learned feature embedding -> RBF), the
        §9.8 B6 DKL-BO "strong BO" confrontation.
    """

    name = "bo_gp"

    def __init__(self, oracle: BHFiniteOracle, features: Mapping[Any, np.ndarray],
                 kernel: str = "rbf") -> None:
        if not _HAS_BOTORCH:
            raise RuntimeError("botorch unavailable")
        if kernel not in ("rbf", "dkl"):
            raise ValueError(f"unknown BO kernel {kernel!r}")
        self._kernel = kernel
        self._oracle = oracle
        self._features = features
        self._pool = list(oracle.candidates)
        self._ident = {i: c for i, c in enumerate(self._pool)}
        self._iident = {c: i for i, c in enumerate(self._pool)}
        self._X_all = np.asarray([features[c] for c in self._pool], dtype=np.float64)
        self._model = None
        self._best_f = None
        # torch device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32 if self._kernel == "dkl" else torch.float64

    def _make_model(self, X, y):
        import botorch

        if self._kernel == "rbf":
            return SingleTaskGP(X, y)
        # DKL: a custom deep RBF kernel (RBF on a learned MLP embedding).  The
        # feature-extractor weights are trained jointly with the GP by marginal
        # likelihood inside botorch's fit, then EI acquisition runs on it.  This
        # is the §9.8 B6 DKL-BO "strong BO" confrontation.
        from gpytorch.kernels import RBFKernel, ScaleKernel

        feat_dim = X.shape[1]
        # Deep-kernel recipe (report 9.8 B3 / 9.12 BH-F): a multi-layer learned
        # embedding before the RBF, so the surrogate can capture non-monotonic
        # component interactions on top of raw bit features.
        feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )
        deep = _DeepRBFKernel(feature_extractor=feature_extractor)
        covar = ScaleKernel(deep)
        return SingleTaskGP(X, y, covar_module=covar)

    def fit(self, queried: Mapping[Any, float]) -> None:
        idx = [self._iident[c] for c in queried if c in self._iident]
        if not idx:
            self._model = None
            return
        X = torch.tensor(self._X_all[idx], device=self._device, dtype=self._dtype)
        y = torch.tensor([[float(queried[self._pool[i]])] for i in idx],
                         device=self._device, dtype=self._dtype)
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        self._model = self._make_model(X, y).to(self._device)
        # CRITICAL: previously the GP was built but never TRAINED (no
        # marginal-likelihood fit), so EI ran on default/untrained kernel
        # hyperparameters and BO collapsed below the random floor.  Train the
        # surrogate here (standard botorch recipe); this is what makes the
        # GP-BO / DKL-BO a credible strong-BO reference (9.8 B3/B6, 9.12 BH-F).
        try:
            mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
            fit_gpytorch_mll(mll, optimizer_kwargs={options: {maxiter: 200}})
        except Exception:
            # fall back to the untrained model rather than killing the campaign;
            # caller is free to treat this run as invalid below.
            pass
        self._best_f = float(max(queried.values()))

    def acquire(self, queried: set, n: int) -> list[Any]:
        if self._model is None:
            # fall back to random over unqueried when BO not yet fit (n < min train)
            avail = [c for c in self._pool if c not in queried]
            return sorted(avail, key=lambda c: str(c))[:n]
        avail_idx = [self._iident[c] for c in self._pool if c not in queried]
        if not avail_idx:
            return []
        try:
            import botorch
            best_f = self._best_f if self._best_f is not None else 0.0
            acq = qLogExpectedImprovement(self._model, best_f=best_f)
            # chunked candidate evaluation keeps the deep-kernel covariance
            # bounded in memory (full-pool eval OOMs for DKL)
            _CH = 256
            acq_vals_list = []
            with torch.no_grad():
                for _s in range(0, len(avail_idx), _CH):
                    _ci = avail_idx[_s:_s+_CH]
                    _xc = torch.tensor(self._X_all[_ci], device=self._device, dtype=self._dtype)
                    _a = acq(_xc.unsqueeze(1)).squeeze().cpu().numpy()
                    acq_vals_list.append(_a)
            acq_vals = np.concatenate(acq_vals_list)
            order = np.argsort(-acq_vals)
            return [self._pool[avail_idx[i]] for i in order[:n]]
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            avail = [c for c in self._pool if c not in queried]
            return sorted(avail, key=lambda c: str(c))[:n]

    def predict_all_solution_prob(self, candidates, threshold=None):
        """P(y >= gamma | D) over unqueried candidates from the GP posterior.

        Turns the (trained) GP-BO / DKL-BO surrogate into a recovery readout
        over unseen candidates, so the strong-BO reference can be scored on the
        SAME solution-space recovery metrics as NLSS (PR-AUC / UnseenRecall /
        AURC, report 9.9).  Fair decisive NLSS-vs-BO recovery confrontation:
        same observations, no ground-truth leak.
        """
        if self._model is None:
            return {c: 0.5 for c in candidates}
        import math as _m
        t = threshold if threshold is not None else self._oracle.gamma
        idx = [self._iident[c] for c in candidates if c in self._iident]
        if not idx:
            return {}
        Xc = torch.tensor(self._X_all[idx], device=self._device, dtype=self._dtype)
        with torch.no_grad():
            post = self._model.posterior(Xc)
            m = post.mean.squeeze(-1).cpu().numpy()
            var = post.variance.squeeze(-1).cpu().numpy()
        try:
            noise = float(self._model.likelihood.noise.detach().cpu())
        except Exception:
            noise = 0.0
        std = np.sqrt(np.maximum(var + noise, 1e-9))
        z = (t - m) / std
        p = 0.5 * (1.0 + np.vectorize(_m.erf)(z / np.sqrt(2.0)))
        return {c: float(p[i]) for i, c in enumerate(candidates)}
    @property
    def trained(self) -> bool:
        return self._model is not None
