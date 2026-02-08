'''
TORCWA Example
Multilayer RCWA Reflection spectrum
'''

# =========================================================
# Import
# =========================================================
import numpy as np
import torch
from matplotlib import pyplot as plt

import torcwa
import Materials

import tqdm

# =========================================================
# Hardware & precision
# =========================================================
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cpu') 

# =========================================================
# Simulation environment
# =========================================================
# light
inc_ang = 0. * (np.pi/180)
azi_ang = 0. * (np.pi/180)

# material (input medium: glass)
substrate_eps = 1.46**2

# =========================================================
# Geometry
# =========================================================
L = [300., 300.]    # period (nm)

torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = L[0]
torcwa.rcwa_geo.Ly = L[1]
torcwa.rcwa_geo.nx = 300
torcwa.rcwa_geo.ny = 300
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 1000.

# z axis (for field view if needed)
z = torch.linspace(-500, 1500, 501, device=device)

x_axis = torcwa.rcwa_geo.x.cpu()
y_axis = torcwa.rcwa_geo.y.cpu()
z_axis = z.cpu()

# =========================================================
# Layer geometry (patterned TiO2)
# =========================================================
layer0_geometry = torcwa.rcwa_geo.rectangle(
    Wx=180., Wy=100.,
    Cx=L[0]/2., Cy=L[1]/2.
)

# =========================================================
# Visualize geometry
# =========================================================
plt.figure()
plt.imshow(
    torch.transpose(layer0_geometry, -2, -1).cpu(),
    origin='lower',
    extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
)
plt.title('Layer 1 geometry (TiO2 in LIC)')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.colorbar()
plt.show()

# =========================================================
# RCWA parameters
# =========================================================
order_N = 15
order = [order_N, order_N]

# wavelength scan (nm)
lamb0 = torch.linspace(400., 700., 21, dtype=geo_dtype, device=device)

# eps_LIC  = 1.55**2
eps_LIC  = 1.55**2
# storage
rxx = []

# =========================================================
# Wavelength loop
# =========================================================
for lamb0_ind in tqdm.tqdm(range(len(lamb0))):

    lamb0_now = lamb0[lamb0_ind]

    # -----------------------------------------------------
    # Initialize RCWA solver
    # -----------------------------------------------------
    sim = torcwa.rcwa(
        freq=1 / lamb0_now,
        order=order,
        L=L,
        dtype=sim_dtype,
        device=device
    )

    # -----------------------------------------------------
    # Input layer: glass
    # -----------------------------------------------------
    sim.add_input_layer(eps=substrate_eps)

    sim.set_incident_angle(
        inc_ang=inc_ang,
        azi_ang=azi_ang
    )

    # -----------------------------------------------------
    # Layer 1: 200 nm TiO2 patterned in LIC
    # -----------------------------------------------------
    eps_TiO2 = Materials.TiO2.apply(lamb0_now)**2


    layer1_eps = (
        layer0_geometry * eps_TiO2 +
        (1. - layer0_geometry) * eps_LIC
    )

    sim.add_layer(
        thickness=200.,
        eps=layer1_eps
    )

    # -----------------------------------------------------
    # Layer 2: 200 nm LIC (uniform)
    # -----------------------------------------------------
    sim.add_layer(
        thickness=200.,
        eps=eps_LIC
    )

    # -----------------------------------------------------
    # Layer 3: 100 nm SiO2 (uniform)
    # -----------------------------------------------------
    eps_SiO2 = 1.46**2

    sim.add_layer(
        thickness=100.,
        eps=eps_SiO2
    )

    # -----------------------------------------------------
    # Layer 4: 100 nm Al (uniform metal)
    # -----------------------------------------------------
    eps_Al = Materials.Al.apply(lamb0_now)**2

    sim.add_layer(
        thickness=100.,
        eps=eps_Al
    )

    # -----------------------------------------------------
    # Solve RCWA
    # -----------------------------------------------------
    sim.solve_global_smatrix()

    # -----------------------------------------------------
    # 0th-order reflection (x → x)
    # -----------------------------------------------------
    rxx.append(
        sim.S_parameters(
            orders=[0, 0],
            direction='backward',
            port='reflection',
            polarization='xx',
            ref_order=[0, 0]
        )
    )

# concatenate
rxx = torch.cat(rxx)

# =========================================================
# Reflection spectrum
# =========================================================
Rxx = torch.abs(rxx)**2

plt.figure()
plt.plot(lamb0.cpu(), Rxx.cpu(), '-o')
plt.title('Reflection spectrum (0th order, xx)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.grid()
plt.savefig('Example1_Reflection_Spectrum.png', dpi=300)
plt.close()
