"""Block 2 refit: delegation 2023-2026 with core-task x year intercepts, so the format change in the task question
cannot masquerade as (or hide) the decay of core protection. Same raw units and pooling as structural.py."""
import sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(f"{sys.argv[1]}/structural.py").read().split('if __name__ == "__main__":')[0])
def block_delegation2(D):
    D = D[(D.uses_ai == 1) & D.task_share.notna()]
    L = D.melt(id_vars=["id", "year", "theta", "sigma"] + CTRL, value_vars=TASKS, var_name="task", value_name="x")
    y = L.x.values; th = L.theta.values; sg = L.sigma.values; Z = L[CTRL].values
    ti = pd.Categorical(L.task, TASKS).codes; yi = np.searchsorted(YEARS, L.year.values)
    core = L.task.map(CORE).fillna(0).values; q = L.year.map(Q).values
    names = [f"a_{t}" for t in TASKS] + [f"d_{t}" for t in YEARS[1:]] + [f"core_x_{t}" for t in YEARS[1:]] + [f"g_{c}" for c in CTRL] + \
            ["kappa_theta", "b_sigma", "kappa0", "rho", "log_sigma0"]
    def ll(b):
        a = b[0:8][ti]; d = np.r_[0, b[8:11]][yi]; cy = np.r_[0, b[11:14]][yi] * core; g = b[14:17]
        kt, bs, k0, rho, ls0 = b[17:22]; st = sg / (sg + np.exp(ls0))
        return ll_bin(y, a + d + cy + Z @ g - kt * th + bs * st - core * k0 * np.exp(-rho * q) * st)
    x0 = np.r_[np.zeros(8), np.zeros(3), np.zeros(3), np.zeros(3), 0.02, 0.0, 0.5, 0.5, np.log(10)]
    return fit_block(ll, x0, names, L.id.astype(str) + L.year.astype(str))
fits = [block_delegation2(E[E.imp == m]) for m in range(1, 6)]
print("converged:", [f[2] for f in fits])
R = rubin(fits); R.insert(0, "block", "2b delegation with core x year"); print(R.round(4).to_string())
R.to_csv(f"{sys.argv[1]}/structural_block2b.csv")
