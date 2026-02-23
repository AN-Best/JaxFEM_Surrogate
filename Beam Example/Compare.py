import meshio
import matplotlib.pyplot as plt
import numpy as np
import os

NN_folder = os.path.join(os.path.dirname(__file__), 'NNTopoOpt_results')
trad_folder = os.path.join(os.path.dirname(__file__), 'TraditionalTopt_data')

mesh_nn = meshio.read(os.path.join(NN_folder, 'vtk/sol_050.vtu'))
theta_nn = mesh_nn.cell_data['theta'][0].reshape(60, 30)

mesh_t = meshio.read(os.path.join(trad_folder, 'vtk/sol_050.vtu'))
theta_t = mesh_t.cell_data['theta'][0].reshape(60, 30)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].imshow((theta_t.T > 0.5).astype(float), cmap='gray_r', origin='lower')
axes[0].set_title('Traditional (FEM-driven)')

axes[1].imshow((theta_nn.T > 0.5).astype(float), cmap='gray_r', origin='lower')
axes[1].set_title('Surrogate-driven')

plt.tight_layout()
plt.show()