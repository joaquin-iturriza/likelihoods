import os
import numpy as np
import glob

# ==== Configurable Parameters ====
base_dir = "/eos/user/j/joiturri/likelihoods"
home_dir = os.path.join(os.path.expanduser("~"),'likelihoods')
# Define the dataset and model
dataset = "1911.12606-EWKinos-1M-z4-nll400-delta200"
model = "mup_mlp"
model_name = 'mup_mlp'
intermediate = False  # Set to True if you want to generate intermediate jobs
# Define directories
# ==== Clean up old job submission files ====

subs_dir = os.path.join(home_dir, "subs")
pattern = os.path.join(subs_dir, f"{dataset}_{model_name}_*_FINAL_l2_lr_grid_small_model_*.sub")

old_subs = glob.glob(pattern)

if old_subs:
    print(f"\n🧹 Found {len(old_subs)} old submission files matching pattern:")
    for f in old_subs:
        print(f"  - {os.path.basename(f)}")
    delete_now = input("Delete these old files before generating new ones? (y/N): ").strip().lower()
    if delete_now == "y":
        for f in old_subs:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not delete {f}: {e}")
        print("Old submission files deleted.")
    else:
        print("Skipping deletion.")
else:
    print(f"\nNo old submission files found in {subs_dir}.")


def to_scientific(num, decimals=2):
    """
    Convert a number to a scientific notation string.
    
    Example:
        to_scientific(1234, decimals=2) -> "1.23e3"
        to_scientific(0.01234, decimals=3) -> "1.234e-2"
    """
    # Use Python's format specification mini-language
    s = f"{num:.{decimals}e}"
    
    # Simplify the exponent (remove leading zeros in exponent)
    base, exp = s.split("e")
    exp = int(exp)  # convert exponent to int to remove + or leading zeros
    
    # Construct the final formatted string
    return f"{base}e{exp}"

def mlp_flops(layer_sizes, batch, count_mac_as_2flops=True, training=True,
              activation_cost_per_el=10, include_bias=True, optimizer="adam"):
    """
    layer_sizes: [d0, d1, ..., dL]
    activation_cost_per_el: e.g., 0 for ReLU, 10 for GELU (rough)
    optimizer: None | 'sgd' | 'adam'
    """
    B = batch
    matmacs = sum(d_in * d_out for d_in, d_out in zip(layer_sizes[:-1], layer_sizes[1:]))
    bias_els = sum(d_out for d_out in layer_sizes[1:])
    act_els  = sum(B * d_out for d_out in layer_sizes[1:])
    macs_fwd = B * matmacs
    if count_mac_as_2flops:
        flops_per_mac = 2
    else:
        flops_per_mac = 1

    # Forward GEMM FLOPs
    flops_fwd = flops_per_mac * macs_fwd
    # Bias + activations
    flops_fwd += (B * bias_els if include_bias else 0) + activation_cost_per_el * act_els

    if not training:
        return {"forward_flops": flops_fwd}

    # Backward: two more GEMMs of same size
    flops_bwd = 2 * flops_per_mac * macs_fwd
    # Activation backward (roughly same order as forward activation)
    flops_bwd += activation_cost_per_el * act_els
    total = flops_fwd + flops_bwd

    # Optimizer cost per step (rough heuristics)
    if optimizer is not None:
        params = matmacs + bias_els
        if optimizer.lower() in ("adam", "adamw"):
            opt_per_param = 10
        elif optimizer.lower() in ("sgd", "momentum"):
            opt_per_param = 3
        else:
            opt_per_param = 2
        total += opt_per_param * params

    return {
        "forward_flops": flops_fwd,
        "train_step_flops": total
    }

# Define the variable parameter(s)
Ns = [10**4]
n_nodess = [64]
n_layerss = [5]
#patiences = [2500000,250000,2500000]
#iterationss = [25000000,2500000,25000000]
flavours = ['testmatch', 'testmatch', 'testmatch']
paramss = np.logspace(1,6,11) 
for dataset in ["1911.12606-EWKinos-1M-z4-nll400-delta200"]:
    for loss in ["HETEROSC"]:
        job_dir = os.path.join(home_dir, f"jobs/{dataset}")
        output_dir = os.path.join(home_dir, f"output/{dataset}/{model}")
        error_dir = os.path.join(home_dir, f"error/{dataset}/{model}")
        log_dir = os.path.join(home_dir, f"log/{dataset}/{model}")
        run_script = os.path.join(base_dir, "run.py")
        # Ensure job directory exists
        for path in [job_dir, output_dir, error_dir, log_dir]:
                os.makedirs(path, exist_ok=True)
        
        for N, n_nodes, n_layers, flavour in zip(Ns, n_nodess, n_layerss, flavours):
            BS=2**11
            if dataset=="qq_tth":
                paramss = np.logspace(-5,1,13) 
                # if N==10**6:
                iterations=[int((x/min(x,(BS)))*10**3) for x in [100000]*len(paramss)]
                # else:
                #     compute_1e6=mlp_flops([500]*4,(2**8))['train_step_flops']*np.array([int((x/min(x,(2**8)))*10**4) for x in np.logspace(1,6,11)], dtype=np.int64)
                #     compute_per_tr_step=mlp_flops([n_nodes]*n_layers,(2**8))['train_step_flops']
                #     iterations=[int(comp_1e6/compute_per_tr_step) for comp_1e6 in compute_1e6]

                amp_trafos="false"
            # elif dataset=="zgggg_10M":
            #     paramss = np.logspace(1,7,13)
            #     if N==10**6:
            #         iterations=[int((x/min(x,(2**8)))*10**4) for x in [10000]]+[int((1e6/min(1e6,(2**8)))*10**4),int((1e6/min(1e6,(2**8)))*10**4)]
            #     else:
            #         compute_1e6=mlp_flops([500]*4,(2**8))['train_step_flops']*np.array([int((x/min(x,(2**8)))*10**4) for x in np.logspace(1,6,11)], dtype=np.int64)
            #         compute_per_tr_step=mlp_flops([n_nodes]*n_layers,(2**8))['train_step_flops']
            #         iterations=[int(comp_1e6/compute_per_tr_step) for comp_1e6 in compute_1e6]+[int((1e6/min(1e6,(2**8)))*10**4),int((1e6/min(1e6,(2**8)))*10**4)]
            #     #iterations=[int((x/min(x,(2**8)))*10**4) for x in np.logspace(1,6,11)]+[int((1e6/min(1e6,(2**8)))*10**4),int((1e6/min(1e6,(2**8)))*10**4)]
            #     amp_trafos='["log","standardization"]'
            else:
                paramss = np.logspace(-5,1,13) 
                # if N==10**6:
                iterations=[int((x/min(x,(BS)))*2*10**3) for x in [100000]*len(paramss)]
                # else:
                #     compute_1e6=mlp_flops([500]*4,(2**8))['train_step_flops']*np.array([int((x/min(x,(2**8)))*10**4) for x in np.logspace(1,6,11)], dtype=np.int64)
                #     compute_per_tr_step=mlp_flops([n_nodes]*n_layers,(2**8))['train_step_flops']
                #     iterations=[int(comp_1e6/compute_per_tr_step) for comp_1e6 in compute_1e6]
                amp_trafos='["standardization"]'
            #LGATR: data.incl_fvs=true data.trafos=false data.amp_trafos=false
            #model.net.hidden_mv_channels=128 model.net.hidden_s_channels=64 model.net.attention.num_heads=16 model.net.num_blocks={bl}
            #MLPI: data.trafos.fvs_standardized=[] data.amp_trafos=false
            #MLPI: data.trafos.fvs_standardized=[] data.amp_trafos=false
            #model.net.hidden_channels=2048 model.net.hidden_layers=6
            # Generate .sh scripts
            #paramss = np.logspace(-3,-1,11) 
            #grids already done
            paramss = [1e-5,3.16e-5,1e-4,3.16e-4,1e-3]
            L2s = [1e-5,3.16e-5,1e-4,3.16e-4,1e-3]

            # paramss2 = [1e-6,3.16e-6,1e-5,3.16e-5,1e-4,3.16e-4,1e-3,3.16e-3,1e-2]
            # L2s2 = [3.16e-7,1e-6,3.16e-6,1e-5,3.16e-5]
            #already done combinations:
            L2s_lrs_done = []
            for lr in paramss:
                for L2 in L2s:
                    L2s_lrs_done.append((L2,lr))

            # for lr in paramss2:
            #     for L2 in L2s2:
            #         L2s_lrs_done.append((L2,lr))
            ############################

            paramss = [1.e-3,3.16e-3,1e-2,3.16e-2]
            L2s = [1e-4,3.16e-4,1e-3,3.16e-3,1e-2]
            job_ids = []
            for L2 in L2s:
                for i, pars in enumerate(paramss, 1):
                    # if (L2,pars) in L2s_lrs_done:
                    #     continue
                    #subsample = int(subsample)
                    params = f"""model={model} training.lr={pars} training.loss={loss} training.batchsize={BS} training.iterations=200000 \
                    data.dataset=[{dataset}] training.scheduler=CosineAnnealingLR training.cosanneal_warmup_steps=4000 training.get_ID=true \
                    training.save_preds_intermediate=true model.net.hidden_channels={n_nodes} model.net.hidden_layers={n_layers} \
                    training.regularization=L2 training.regularization_lambda={L2} \
                    training.es_patience=100000000"""
                    #if subsample=='no_subs':
                    #    exp_name = f"amp_{dataset}_{model_name}_nosubs"
                    #else:
                    exp_name = f"{dataset}_{model_name}_{loss}"
                    #    params += f" data.subsample={subsample}"
                    job_id = f"{i}"
                    sh_path = os.path.join(job_dir, f"job_{dataset}_{model_name}_{loss}_{job_id}_FINAL_l2_lr_grid_small_model_{to_scientific(N,0)}_N_{to_scientific(L2)}_L2.sh")
                    if intermediate:
                        sh_path = sh_path.replace(".sh", "_intermediate.sh")
                        exp_name += "_intermediate"
                        params += ' training.save_intermediate=true'
                    params += f" exp_name={exp_name}_FINAL_{to_scientific(L2)}_{to_scientific(pars)}_l2_lr_grid_small_model_{to_scientific(N,0)}_N"
                    with open(sh_path, "w") as f:
                        f.write(f"""#!/bin/bash
                            cd {base_dir}
                            source ../jitu/amplitude_DSI/amplitudes_env/bin/activate
                            python {run_script} {params}
                            """)
                    os.chmod(sh_path, 0o755)
                    job_ids.append(job_id)

                    # Generate the .sub submission script
                    if iterations[i-1]<1e5:
                        flavour='tomorrow'
                    elif iterations[i-1]<4e5:
                        flavour='tomorrow'
                    elif iterations[i-1]<1.6e6:
                        flavour='tomorrow'
                    elif iterations[i-1]<5e6:
                        flavour='tomorrow'
                    elif iterations[i-1]<1.2e7:
                        flavour='testmatch'
                    else:
                        flavour='nextweek'

                    sub_path = os.path.join(home_dir, f"subs/{dataset}_{model_name}_{loss}_FINAL_l2_lr_grid_small_model_{to_scientific(N,0)}_N_{to_scientific(L2)}_L2_{i}.sub")
                    job_name = f"job_{dataset}_{model_name}_{loss}_$(ID)_FINAL_l2_lr_grid_small_model_{to_scientific(N,0)}_N_{to_scientific(L2)}_L2.sh"
                    if intermediate:
                        job_name = job_name.replace(".sh", "_intermediate.sh")
                        sub_path = sub_path.replace(".sub", "_intermediate.sub")

                    #if model_name == 'mlpi':
                    #    flavour = "testmatch"
                    #elif model_name == 'lgatr':
                    #    flavour = "testmatch"
                    with open(sub_path, "w") as f: #"testmatch" "workday" "longlunch" "tomorrow" "nextweek" "espresso"
                        f.write(f"""executable            = {job_dir}/{job_name}
                    arguments             = $(ClusterId) $(ProcId)
                    output                = {output_dir}/{job_name}.$(ClusterId).$(ProcId).out
                    error                 = {error_dir}/{job_name}.$(ClusterId).$(ProcId).err
                    log                   = {log_dir}/{job_name}.$(ClusterId).$(ProcId).log
                    request_gpus          = 1
                    requirements          = !regexp("MIG", TARGET.GPUs_DeviceName)
                    +JobFlavour           = "{flavour}"
                    queue ID from (
                    """)
                        #for job_id in job_ids:
                        f.write(f"{i}\n")
                        f.write(")\n")

# Run jobs:

import subprocess

# ==== Automatic submission step ====
subs_dir = os.path.join(home_dir, "subs")
pattern = os.path.join(subs_dir, f"{dataset}_{model_name}_*_FINAL_l2_lr_grid_small_model_*.sub")

# Find all matching submission files
matching_subs = sorted(glob.glob(pattern))

if not matching_subs:
    print(f"\nNo submission files found matching pattern:\n{pattern}")
else:
    print(f"\n>>> Found {len(matching_subs)} submission files:")
    for f in matching_subs:
        print(f"  - {os.path.basename(f)}")

    submit_now = input("\nSubmit all these jobs to HTCondor now? (y/N): ").strip().lower()
    if submit_now == "y":
        submitted = 0
        for sub_path in matching_subs:
            print(f"\nSubmitting {sub_path} ...")
            try:
                subprocess.run(["condor_submit", sub_path], check=True)
                submitted += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit {sub_path}: {e}")
        print(f"\nSubmitted {submitted} jobs successfully.")
    else:
        print("\nSkipping job submission. You can submit later manually with:")
        print(f"cd {subs_dir} && for f in {os.path.basename(pattern)}; do condor_submit $f; done")

# ==== End of Script ====