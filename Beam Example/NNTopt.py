import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import torch
import torch.nn as nn

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.mma import optimize

# ─────────────────────────────────────────────
# Surrogate model definition (must match training)
# ─────────────────────────────────────────────
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (16, 30, 15)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 15, 7)
            nn.Flatten()                                  # (32*15*7,)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*15*7 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, theta, x_load):
        features = self.cnn(theta)
        x = torch.cat([features, x_load.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze(1)


# ─────────────────────────────────────────────
# Load surrogate
# ─────────────────────────────────────────────
surrogate_path = os.path.join(os.path.dirname(__file__), 'surrogate.pt')
checkpoint = torch.load(surrogate_path)
surrogate = SurrogateModel()
surrogate.load_state_dict(checkpoint['model'])
surrogate.eval()
y_mean = float(checkpoint['y_mean'])
y_std = float(checkpoint['y_std'])


def surrogate_objective(rho, x_load_val):
    """Wraps PyTorch surrogate to return (J, dJ/drho) as JAX arrays."""
    rho_np = onp.array(rho)                                      # (1800, 1)
    rho_tensor = torch.tensor(rho_np, dtype=torch.float32)
    rho_tensor = rho_tensor.reshape(1, 1, 60, 30).requires_grad_(True)
    x_load_tensor = torch.tensor([x_load_val], dtype=torch.float32)

    # Forward pass
    J_norm = surrogate(rho_tensor, x_load_tensor)

    # Backward pass
    J_norm.backward()

    # Denormalize compliance
    J = float(J_norm.item() * y_std + y_mean)

    # Denormalize gradient via chain rule
    dJ = rho_tensor.grad.reshape(1800, 1).detach().numpy()
    dJ = dJ * y_std

    return np.float32(J), np.array(dJ)


# ─────────────────────────────────────────────
# Mesh
# ─────────────────────────────────────────────
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.
meshio_mesh = rectangle_mesh(Nx=60, Ny=30, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# ─────────────────────────────────────────────
# FEM problem (needed for mesh/output only — not for objective)
# ─────────────────────────────────────────────
class Elasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def stress(u_grad, theta):
            Emax = 70.e3
            Emin = 1e-3*Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*theta[0]**penal
            epsilon = 0.5*(u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22)
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 100.])
        return [surface_map]

    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


# ─────────────────────────────────────────────
# Run surrogate-driven topology optimization
# ─────────────────────────────────────────────
x_load = 60.0  # change this to test different load locations

def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)

def dirichlet_val(point):
    return 0.

def make_load_location(x_load):
    def load_location(point):
        return np.logical_and(
            np.logical_and(point[0] >= x_load - 1.0, point[0] <= x_load + 1.0),
            np.isclose(point[1], 0., atol=1e-5)
        )
    return load_location

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
location_fns = [make_load_location(x_load)]

problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type,
                     dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)

# Output directory
data_path = os.path.join(os.path.dirname(__file__), 'NNTopoOpt_results')
os.makedirs(os.path.join(data_path, 'vtk'), exist_ok=True)

compliance_log = []

def output_sol(params, obj_val):
    # We still need a FEM solve to visualize — use the real solver here
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}},
                          adjoint_solver_options={'umfpack_solver': {}})
    sol_list = fwd_pred(params)
    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path,
             cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"compliance = {obj_val:.4f}")
    output_sol.counter += 1
output_sol.counter = 0


def objectiveHandle(rho):
    J, dJ = surrogate_objective(rho, x_load)
    compliance_log.append(float(J))
    output_sol(rho, J)
    return J, dJ


def consHandle(rho, epoch):
    def computeGlobalVolumeConstraint(rho):
        g = np.mean(rho)/vf - 1.
        return g
    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc


vf = 0.5
optimizationParams = {'maxIters': 51, 'movelimit': 0.1}
rho_ini = vf * np.ones((len(problem.fe.flex_inds), 1))
numConstraints = 1

optimize(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints)

onp.save(os.path.join(data_path, 'compliance_log.npy'), onp.array(compliance_log))
print(f"Done. Final compliance: {compliance_log[-1]:.4f}")