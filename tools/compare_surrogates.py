"""Sampler test: DyHPO (deep-kernel GP) vs plain GP (Matern-5/2 ARD + noise, EI) vs random
order, on synthetic objectives over the rafalmerged_sleptons_002 search space. Same candidate
pool and start-up trials for all three, waves as in a real sweep, diverged trials = worst
value seen. CPU only; run from the checkout (`scripts/compare_surrogates.sh` as a CPU job).
Env: SHAPE in {bowl, edge, real}, NOISE (relative, e.g. 0.01), SEEDS (e.g. 0,1,2), POOL (candidates, default 300).
"real" is fitted by eye to the rafalmerged_sleptons_002 results: flat in L2 below ~1e-4, a narrow optimum
at L2 ~ 1e-3 whose depth grows with lr, over-regularised above ~4e-3, divergence at lr > 2.2e-2 with small L2."""
import sys, os, tempfile, shutil, warnings, numpy as np, yaml, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from sweep.dyhpo_sampler import DyHPOSampler, _preprocess_candidates
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm
torch.set_num_threads(2)
SHAPE, NOISE = os.environ["SHAPE"], float(os.environ["NOISE"])
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
POOL = int(os.environ.get("POOL", "300"))
WAVES = [int(x) for x in os.environ.get("WAVES", "10,5,5,5,5").split(",")]
space = yaml.safe_load(open("sweep/sweep_config_rafalmerged_sleptons_002.yaml"))["search_space"]

def truth(p):
    """Noise-free objective (lower is better), None = diverged."""
    lr, l2, wu, a = p["training.lr"], p["training.regularization_lambda"], p["training.cosanneal_warmup_frac"], p["data.nLL_trafos.asinh_scale"]
    if SHAPE == "real":
        if (lr > 0.022 and l2 < 1e-4) or (lr > 0.0155 and l2 < 1e-7): return None
        base = 1.1e-3 * (1 + 0.17 * max(0.0, np.log10(1e-2 / lr))**2)
        x = np.log10(l2)
        depth = 0.48 * min(1.0, max(0.2, (np.log10(lr) + 3.3) / 1.3))
        g = 1 - depth * np.exp(-0.5 * ((x + 3) / 0.4)**2) + 1.5 * max(0.0, x + 2.4)
        return base * g * (1 + 10 * (wu - 0.07)**2)
    if lr > 0.022: return None
    u = np.log10(lr / 1e-2)
    if SHAPE == "bowl":      # optimum inside the box
        v = np.log10(l2 / 1e-4)
        return 6e-4 * (1 + 0.8 * u**2 + 0.05 * v**2 + 20 * (wu - 0.05)**2)
    # "edge": like the sleptons data, best at large L2 (upper edge) and lr close to divergence
    v = np.log10(l2 / 1e-2)
    return 6e-4 * (1 + 0.8 * (np.log10(lr / 1.8e-2))**2 + 0.12 * v**2 + 0.1 * np.log10(a / 0.3)**2)

def noisy(y, rng):
    return None if y is None else y * (1 + NOISE * rng.standard_normal())

class GPSampler:
    """Plain GP + EI on the same candidates and start-up order as DyHPO (seeded the same)."""
    def __init__(self, ref):
        self.raw = ref.candidates_raw
        self.X = _preprocess_candidates(ref.candidates_array, ref.log_indicator)
        self.init = [int(i) for i in ref.algorithm.init_conf_indices]
        self.obs, self.div, self.flight, self.k = {}, set(), set(), 0
    def suggest(self):
        if self.k < len(self.init):
            h = self.init[self.k]; self.k += 1
        else:
            hs = list(self.obs); y = -np.array([self.obs[h] for h in hs])
            mu, sd = y.mean(), y.std() or 1.0
            ys = list((y - mu) / sd); Xs = [self.X[h] for h in hs]
            for h in self.div: Xs.append(self.X[h]); ys.append(min(ys))
            gp = GPR(C(1.0) * Matern(length_scale=np.ones(self.X.shape[1]), nu=2.5, length_scale_bounds=(1e-2, 1e2))
                     + WhiteKernel(0.1, (1e-4, 1)), n_restarts_optimizer=5, random_state=0).fit(np.array(Xs), np.array(ys))
            cand = [i for i in range(len(self.X)) if i not in self.obs and i not in self.div and i not in self.flight]
            m, s = gp.predict(self.X[cand], return_std=True)
            best = max(ys[:len(hs)]); z = (m - best) / np.maximum(s, 1e-12)
            h = cand[int(np.argmax((m - best) * norm.cdf(z) + s * norm.pdf(z)))]
        self.flight.add(h); return h
    def observe(self, h, y): self.flight.discard(h); self.obs[h] = y
    def fail(self, h): self.flight.discard(h); self.div.add(h)

tmp = tempfile.mkdtemp(prefix="zz_cmp_")
def cycle(s):
    p = os.path.join(tmp, "state.pkl"); s.save(p); return DyHPOSampler.load(p, tmp, force_cpu=True)

def run(kind, seed):
    rng = np.random.default_rng(1000 + seed)
    s = DyHPOSampler(space, {"t_steps": [200000]}, POOL, 100 + seed, tmp, 10); s = cycle(s)
    g = GPSampler(s) if kind == "gp" else None
    perm = list(np.random.default_rng(seed).permutation(POOL))
    if kind == "random":     # same start-up trials, then a random order
        init = [int(i) for i in s.algorithm.init_conf_indices]
        perm = init + [i for i in perm if i not in init]
    picks, seen = [], {}
    for wave in WAVES:
        batch = []
        for _ in range(wave):
            if kind == "dyhpo": h, p, t = s.suggest(); s = cycle(s)
            elif kind == "gp": h = g.suggest()
            else: h = perm[len(picks) + len(batch)]
            batch.append(h)
        for h in batch:
            y = noisy(truth(s.candidates_raw[h]), rng); picks.append(h); seen[h] = y
            if kind == "dyhpo":
                if y is None: s.report_failure(h)
                else: s.observe(h, 200000, y)
                s = cycle(s)
            elif kind == "gp":
                g.fail(h) if y is None else g.observe(h, y)
    ok = [h for h in picks if seen[h] is not None]
    chosen = min(ok, key=lambda h: seen[h])            # what the sweep would select (noisy best)
    post = picks[10:]
    pt = [truth(s.candidates_raw[h]) for h in post]
    return dict(chosen=truth(s.candidates_raw[chosen]),
                best_true=min(truth(s.candidates_raw[h]) for h in ok),
                post_med=float(np.median([v for v in pt if v is not None])),
                div=sum(v is None for v in pt) / len(pt))

try:
    pools = [DyHPOSampler(space, {"t_steps": [200000]}, POOL, 100 + sd, tmp, 10).candidates_raw for sd in SEEDS]
    opts = [min(v for v in map(truth, c) if v is not None) for c in pools]
    print(f"SHAPE={SHAPE} NOISE={NOISE} POOL={POOL}: best candidate in each seed's pool (true) " + " ".join(f"{o:.3e}" for o in opts), flush=True)
    res = {k: [] for k in ["dyhpo", "gp", "random"]}
    for seed in SEEDS:
        for k in res:
            r = run(k, seed); res[k].append(r)
            print(f"  seed {seed} {k:6s}: selected (true) {r['chosen']:.3e}  best tried {r['best_true']:.3e}  picks 11-30 median {r['post_med']:.3e}  diverged {100*r['div']:.0f}%", flush=True)
    for k, rs in res.items():
        print(f"MEAN {SHAPE} {NOISE} {POOL} {k:6s}: selected {np.mean([r['chosen'] for r in rs]):.3e}  best tried {np.mean([r['best_true'] for r in rs]):.3e}  "
              f"picks 11-30 median {np.mean([r['post_med'] for r in rs]):.3e}  diverged {100*np.mean([r['div'] for r in rs]):.0f}%", flush=True)
finally:
    shutil.rmtree(tmp)
