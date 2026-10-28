"""Latent class analysis (mixture of independent categorical variables), fitted by EM with multiple starts."""
import numpy as np
def onehot(X, levels):
    # a missing answer (NaN) gives an all-zero row, so it adds nothing to the likelihood (full-information under MAR)
    return [np.stack([(X[:, j] == v).astype(float) for v in levels[j]], 1) for j in range(X.shape[1])]
def fit(X, levels, k, starts=8, iters=400, tol=1e-7, seed=0):
    rng = np.random.default_rng(seed); OH = onehot(X, levels); n = X.shape[0]; best = None
    for s in range(starts):
        R = rng.dirichlet(np.ones(k), n); ll_old = -np.inf
        for it in range(iters):
            pi = R.mean(0) + 1e-12
            th = [(R.T @ oh + 1e-3) / ((R * oh.sum(1, keepdims=True)).sum(0)[:, None] + 1e-3 * oh.shape[1]) for oh in OH]   # k x levels
            L = np.log(pi)[None, :] + sum(oh @ np.log(t).T for oh, t in zip(OH, th))
            m = L.max(1, keepdims=True); ll = (m[:, 0] + np.log(np.exp(L - m).sum(1))).sum()
            R = np.exp(L - m); R /= R.sum(1, keepdims=True)
            if ll - ll_old < tol * abs(ll): break
            ll_old = ll
        if best is None or ll > best["ll"]:
            npar = (k - 1) + k * sum(len(l) - 1 for l in levels)
            ent = -(R * np.log(R + 1e-12)).sum()
            best = dict(ll=ll, pi=pi, theta=th, post=R, k=k, bic=-2 * ll + npar * np.log(n),
                        entropy=1 - ent / (n * np.log(k)), npar=npar)
    return best
def predict(X, levels, model):
    OH = onehot(X, levels)
    L = np.log(model["pi"])[None, :] + sum(oh @ np.log(t).T for oh, t in zip(OH, model["theta"]))
    L -= L.max(1, keepdims=True); R = np.exp(L); return R / R.sum(1, keepdims=True)
