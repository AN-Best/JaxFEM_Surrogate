import meshio
import numpy as np
import os

main_folder = os.path.join(os.path.dirname(__file__), 'DataSet_data')

n_runs = 59
n_iter = 51

X_theta = []
X_load = []
y_compliance = []

for i in range(n_runs):
    run_folder = os.path.join(main_folder, f'run_{i:03d}')
    
    x_load = np.load(os.path.join(run_folder, 'x_load.npy'))
    compliance_log = np.load(os.path.join(run_folder, 'compliance_log.npy'))
    
    for t in range(n_iter):
        vtu_path = os.path.join(run_folder, f'vtk/sol_{t:03d}.vtu')
        mesh = meshio.read(vtu_path)
        theta = mesh.cell_data['theta'][0].reshape(60, 30)
        
        X_theta.append(theta)
        X_load.append(float(x_load))
        y_compliance.append(compliance_log[t])

np.savez(os.path.join(main_folder, 'dataset.npz'),
         X_theta=np.array(X_theta),
         X_load=np.array(X_load),
         y_compliance=np.array(y_compliance))

print(f"Dataset saved: {len(y_compliance)} samples")