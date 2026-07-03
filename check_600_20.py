import numpy as np, json, os, onnx, onnxruntime

BASE = os.path.dirname(os.path.abspath(__file__))

# --- Load yields and find closest points ---
with open(f"{BASE}/yields_600_600_20_20.json") as f:
    d = json.load(f)
onshell  = next(x for x in d if x.get("model") == "onshell.onnx")
target   = np.array(onshell["yields"])

data     = np.load(f"{BASE}/data/2106.01676-onshell-winobino-fluct25_-300k.npy", mmap_mode="r")
features = data[:, :-8]
nlls     = data[:, -8:]  # [nll0_exp, nll1_exp, nll0_obs, nll1_obs, nll0_expA, nll1_expA, nll0_obsA, nll1_obsA]

diffs = np.linalg.norm(features - target, axis=1)
top5  = np.argsort(diffs)[:5]

# --- Set up NN ---
onnx_path = f"{BASE}/models_onnx/SUSY-2019-09_Onshell_Winobino.onnx"
m         = onnx.load(onnx_path)
md        = {p.key: p.value for p in m.metadata_props}
std_data  = json.loads(md["standardization"])
feat_mean = np.array(std_data["features_mean"][0])
feat_std  = np.array(std_data["features_std"][0])
nll_mean  = np.array(std_data["nLLs_mean"][0])
nll_std   = np.array(std_data["nLLs_std"][0])
nll0_exp  = json.loads(md["nLL_exp_mu0"])
nll0_obs  = json.loads(md["nLL_obs_mu0"])

def log_w_neg(x):  return np.sign(x) * np.log1p(np.abs(x))
def ilog_w_neg(x): return np.sign(x) * np.expm1(np.abs(x))

sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def predict(yields):
    x = log_w_neg(np.array(yields, dtype=np.float64))
    x = ((x - feat_mean) / feat_std).astype(np.float32).reshape(1, -1)
    raw = sess.run(None, {"features": x})[0][0]
    deltas = ilog_w_neg(raw[:4].astype(np.float64) * nll_std + nll_mean)
    return nll0_exp + deltas[0], nll0_obs + deltas[1]

# --- Print results ---
print(f"{'idx':>8}  {'dist':>7}  {'truth_obs':>10}  {'truth_exp':>10}  {'pred_obs':>10}  {'pred_exp':>10}")
for i in top5:
    p_exp, p_obs = predict(features[i])
    print(f"{i:>8}  {diffs[i]:>7.4f}  {nlls[i,3]:>10.4f}  {nlls[i,1]:>10.4f}  {p_obs:>10.4f}  {p_exp:>10.4f}")

# Also show prediction for the original target yields
p_exp, p_obs = predict(target)
print(f"\nOriginal point yields:")
print(f"  pred_obs={p_obs:.4f}  pred_exp={p_exp:.4f}")
