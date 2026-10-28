"""Fit the identity-assimilation optimisation model by maximum likelihood (random-utility form).
Raw units: investment theta and security sigma in years (no standardisation).
EU professional developers, 2023-2026 (2026 = 70% exploration file). 5 MICE imputations, Rubin's rules,
standard errors clustered by respondent (sandwich with numerical scores).
Usage: python3 -I structural.py <prep_dir>"""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
P = sys.argv[1]
E = pd.read_pickle(f"{P}/encoded_long.pkl")
E = E[E.eu == 1].copy()
E["grad"] = (E.edu_ord >= 6).astype(float); E["large"] = (E.org_ord >= 6).astype(float)
CTRL = ["grad", "large", "manager"]
YEARS = [2023, 2024, 2025, 2026]; Q = {2023: 0.0, 2024: 1.0, 2025: 2.0, 2026: 3.0}   # capability index q_t
TASKS = ["t_writing_code", "t_debugging", "t_reviewing_code", "t_documenting_code", "t_testing_code",
         "t_learning_a_codebase", "t_project_planning", "t_deployment_and_monitoring"]
CORE = {"t_writing_code": 1, "t_debugging": 1, "t_reviewing_code": 1}
def lam(x): return 1 / (1 + np.exp(-np.clip(x, -35, 35)))
def ll_bin(y, eta): p = np.clip(lam(eta), 1e-12, 1 - 1e-12); return y * np.log(p) + (1 - y) * np.log(1 - p)

def fit_block(llfun, x0, names, groups):
    f = lambda b: -llfun(b).sum()
    r = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 5000, "maxfun": 200000})
    b = r.x; h = 1e-5; k = len(b)
    S = np.zeros((len(llfun(b)), k))                     # per-observation scores
    for i in range(k):
        e = np.zeros(k); e[i] = h; S[:, i] = (llfun(b + e) - llfun(b - e)) / (2 * h)
    H = np.zeros((k, k))   # Hessian of the total log-likelihood, by central differences of the gradient
    for i in range(k):
        e = np.zeros(k); e[i] = h
        gp = np.array([(llfun(b + e + ee) - llfun(b + e - ee)).sum() / (2 * h) for ee in np.eye(k) * h])
        gm = np.array([(llfun(b - e + ee) - llfun(b - e - ee)).sum() / (2 * h) for ee in np.eye(k) * h])
        H[:, i] = (gp - gm) / (2 * h)
    Hi = np.linalg.inv(-H)
    Sg = pd.DataFrame(S).groupby(np.asarray(groups)).sum().values; G = Sg.shape[0]
    V = Hi @ (Sg.T @ Sg) @ Hi * G / (G - 1)
    return dict(zip(names, b)), dict(zip(names, np.sqrt(np.clip(np.diag(V), 0, None)))), r.success, -r.fun

def block_participation(D):
    """A = 1[(alpha_t + z'g - phi*theta - psi_t*sigma)/s_t + e]; s_2023 = 1. Tests whether the investment barrier phi
    is constant while the noise scale s_t shrinks."""
    D = D.dropna(subset=["uses_ai"]); y = D.uses_ai.values; th = D.theta.values; sg = D.sigma.values
    Z = D[CTRL].values; yi = np.searchsorted(YEARS, D.year.values)
    names = [f"alpha_{t}" for t in YEARS] + [f"psi_{t}" for t in YEARS] + ["phi"] + [f"g_{c}" for c in CTRL] + [f"logscale_{t}" for t in YEARS[1:]]
    def ll(b):
        a = b[0:4][yi]; psi = b[4:8][yi]; phi = b[8]; g = b[9:12]; ls = np.r_[0, b[12:15]][yi]
        return ll_bin(y, (a + Z @ g - phi * th - psi * sg) / np.exp(ls))
    return fit_block(ll, np.r_[np.ones(4), np.zeros(4), 0.05, np.zeros(3), np.zeros(3)], names, D.id.astype(str) + D.year.astype(str))

def block_delegation(D):
    """x_j = 1[a_j + d_t + z'g - kappa_theta*theta + b_sig*st - s_j*kappa0*exp(-rho*q_t)*st + e],
    st = sigma/(sigma+sigma0). AI users with task answers, 8 shared tasks."""
    D = D[(D.uses_ai == 1) & D.task_share.notna()]
    L = D.melt(id_vars=["id", "year", "theta", "sigma"] + CTRL, value_vars=TASKS, var_name="task", value_name="x")
    y = L.x.values; th = L.theta.values; sg = L.sigma.values; Z = L[CTRL].values
    ti = pd.Categorical(L.task, TASKS).codes; yi = np.searchsorted(YEARS, L.year.values)
    core = L.task.map(CORE).fillna(0).values; q = L.year.map(Q).values
    names = [f"a_{t}" for t in TASKS] + [f"d_{t}" for t in YEARS[1:]] + [f"g_{c}" for c in CTRL] + \
            ["kappa_theta", "b_sigma", "kappa0", "rho", "log_sigma0"]
    def ll(b):
        a = b[0:8][ti]; d = np.r_[0, b[8:11]][yi]; g = b[11:14]
        kt, bs, k0, rho, ls0 = b[14:19]; st = sg / (sg + np.exp(ls0))
        return ll_bin(y, a + d + Z @ g - kt * th + bs * st - core * k0 * np.exp(-rho * q) * st)
    x0 = np.r_[np.zeros(8), np.zeros(3), np.zeros(3), 0.02, 0.0, 0.5, 0.5, np.log(10)]
    return fit_block(ll, x0, names, L.id.astype(str) + L.year.astype(str))

T4 = {"t_writing_code": ("useful_impl", "sat_impl", 1), "t_debugging": ("useful_debug", "sat_debug", 1),
      "t_reviewing_code": ("useful_review", "sat_review", 1), "t_documenting_code": ("useful_docs", "sat_docs", 0)}
def long2026(D):
    D = D[(D.year == 2026) & (D.uses_ai == 1) & D.task_share.notna()]
    rows = []
    for t, (u, s, c) in T4.items():
        rows.append(pd.DataFrame({"id": D.id.values, "x": D[t].values, "g": (D[u] >= 3).astype(float).where(D[u].notna()).values,
                                  "satdown": (D[s] <= 2).astype(float).where(D[s].notna()).values, "core": float(c),
                                  "theta": D.theta.values, "sigma": D.sigma.values, "p": D.pressure_ord.values,
                                  **{c_: D[c_].values for c_ in CTRL}}))
    return pd.concat(rows, ignore_index=True)

def block_gain_pressure(D):
    """2026: x_j = 1[c0 + c_core*s_j + beta_g*g + beta_pm*med + beta_ph*high - kappa_theta*theta + b_sigma*sigma + z'g + e]"""
    L = long2026(D).dropna(subset=["x", "g", "p"]); y = L.x.values; Z = L[CTRL].values
    med = (L.p == 2).astype(float).values; hi = (L.p == 3).astype(float).values
    names = ["c0", "c_core", "beta_g", "beta_p_medium", "beta_p_high", "kappa_theta", "b_sigma"] + [f"g_{c}" for c in CTRL]
    def ll(b):
        return ll_bin(y, b[0] + b[1] * L.core.values + b[2] * L.g.values + b[3] * med + b[4] * hi - b[5] * L.theta.values + b[6] * L.sigma.values + Z @ b[7:10])
    return fit_block(ll, np.zeros(10), names, L.id.astype(str))

def block_felt(D):
    """2026: Pr(AI lowered satisfaction) = Lambda(c0 + lambda*s_j*(1-st) - eta*st + mu*theta + pi_m*med + pi_h*high - gamma_g*g + z'g),
    st = sigma/(sigma+sigma0f)."""
    L = long2026(D).dropna(subset=["satdown", "g", "p"]); y = L.satdown.values; Z = L[CTRL].values
    med = (L.p == 2).astype(float).values; hi = (L.p == 3).astype(float).values
    names = ["c0", "lambda", "eta", "mu", "pi_medium", "pi_high", "gamma_g"] + [f"g_{c}" for c in CTRL] + ["log_sigma0_felt"]
    def ll(b):
        st = L.sigma.values / (L.sigma.values + np.exp(b[10]))
        return ll_bin(y, b[0] + b[1] * L.core.values * (1 - st) - b[2] * st + b[3] * L.theta.values + b[4] * med + b[5] * hi - b[6] * L.g.values + Z @ b[7:10])
    return fit_block(ll, np.r_[np.zeros(10), np.log(10)], names, L.id.astype(str))

def block_verify(D):
    """2026, among those delegating writing code or debugging: own-expertise checking = Lambda(w0 + nu*theta + b*sigma + z'g);
    outside-source checking = Lambda(v0 + omega1*(1 - st) + b*theta + z'g), st = sigma/(sigma+sigma0v)."""
    D = D[(D.year == 2026) & ((D.t_writing_code == 1) | (D.t_debugging == 1))]
    A_ = D.dropna(subset=["verify_own"]); B_ = D.dropna(subset=["verify_external"])
    Za, Zb = A_[CTRL].values, B_[CTRL].values
    names = ["w0", "nu_theta", "b_sigma_own"] + [f"ga_{c}" for c in CTRL] + ["v0", "omega1", "b_theta_ext"] + [f"gb_{c}" for c in CTRL] + ["log_sigma0_verify"]
    def ll(b):
        la = ll_bin(A_.verify_own.values, b[0] + b[1] * A_.theta.values + b[2] * A_.sigma.values + Za @ b[3:6])
        st = B_.sigma.values / (B_.sigma.values + np.exp(b[12]))
        lb = ll_bin(B_.verify_external.values, b[6] + b[7] * (1 - st) + b[8] * B_.theta.values + Zb @ b[9:12])
        return np.r_[la, lb]
    grp = np.r_[A_.id.astype(str).values, B_.id.astype(str).values]
    return fit_block(ll, np.r_[np.zeros(12), np.log(10)], names, grp)

def rubin(fits):
    names = list(fits[0][0]); m = len(fits)
    est = np.array([[f[0][n] for n in names] for f in fits]); se = np.array([[f[1][n] for n in names] for f in fits])
    q = est.mean(0); W = (se ** 2).mean(0); B = est.var(0, ddof=1); T = np.sqrt(W + (1 + 1 / m) * B)
    return pd.DataFrame({"estimate": q, "se": T, "z": q / T}, index=names)

if __name__ == "__main__":
    blocks = {"1 participation": block_participation, "2 delegation 2023-2026": block_delegation,
              "3 gain and pressure 2026": block_gain_pressure, "4 felt utility 2026": block_felt, "5 verification 2026": block_verify}
    out = []
    for name, fn in blocks.items():
        fits = []
        for m in range(1, 6):
            fits.append(fn(E[E.imp == m])); print(name, "imp", m, "converged", fits[-1][2], flush=True)
        R = rubin(fits); R.insert(0, "block", name); out.append(R)
        print(R.round(4).to_string(), flush=True)
    pd.concat(out).to_csv(f"{P}/structural_estimates.csv")
