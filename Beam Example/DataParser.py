import numpy as np
import os

main_folder = os.path.join(os.path.dirname(__file__), 'DataSet_data')

n_runs = 295
n_iter = 51

X_theta = []
X_load = []
X_vf = []
y_compliance = []

for i in range(n_runs):
    run_folder = os.path.join(main_folder, f'run_{i:04d}')

    if not os.path.exists(run_folder):
        print(f"Warning: {run_folder} not found, skipping")
        continue

    x_load = float(np.load(os.path.join(run_folder, 'x_load.npy')))
    vf = float(np.load(os.path.join(run_folder, 'vf.npy')))
    compliance_log = np.load(os.path.join(run_folder, 'compliance_log.npy'))
    theta_log = np.load(os.path.join(run_folder, 'theta_log.npy'))  # (51, 60, 30)

    for t in range(n_iter):
        X_theta.append(theta_log[t])
        X_load.append(x_load)
        X_vf.append(vf)
        y_compliance.append(compliance_log[t])

np.savez(os.path.join(main_folder, 'dataset.npz'),
         X_theta=np.array(X_theta),
         X_load=np.array(X_load),
         X_vf=np.array(X_vf),
         y_compliance=np.array(y_compliance))

print(f"Dataset saved: {len(y_compliance)} samples")