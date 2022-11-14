import copy

import matplotlib.pyplot as plt
import meshio
import numpy as np
import sys
sys.path.append("../scripts")
from utils import plot_mesh, extract_grid, transform_mesh, choose_rbf, save_mesh, check_validity, stretch_cp, rotate_cp

name = 'WS'
msh_template = meshio.read('LAA_template.msh')
template_points, template_cells, template_border = extract_grid(msh_template, flag_border=True)
radial_basis = choose_rbf(name='l2')
np.random.seed(2304)

# control points
coordinates_init = np.array([
                             [1, 1.05],
                             [2.1, -0.15],
                             [2, -0.35],
                             [0.75, 0.15],
                             [2.2, -0.35], # extreme point
                             [0.37784, 0.12655],
                             [1.14838, 0.01569],
                             [1.69354, -0.23087],
                             [1.71713, 0.34794],
                             [1.31782, 0.79320],
                             [0.29379, 1.05358],
                             [0.13842, 1.01739],
                             [0.68486, 1.12679],
                             [1.80490, 0.23936],
                             [1.46963, 0.63484],
                             [1.97590, 0.01863]
                             ])

xy = np.array([np.zeros(shape=4), np.linspace(0,1,4)]).T
coordinates_init = np.concatenate([xy, coordinates_init])
n_control_points = coordinates_init.shape[0]

# control points displacements
coordinates_final = np.array([
                             [1.1, 0.9],
                             [2.8, 0.1],
                             [2.4, -0.03],
                             [0.9, 0.2],
                             [3, -0.1],
                             [0.3, 0.25],
                             [1.55, 0.12],
                             [1.95, 0.05],
                             [1.98465, 0.32270],
                             [1.6, 1.05],
                             [0.36, 1],
                             [0.20, 0.93],
                             [0.87335, 1.05],
                             [2.11194, 0.26577],
                             [1.8, 0.8],
                             [2.47255, 0.20377]
                             ])

coordinates_final = np.concatenate([xy, coordinates_final])

#parameter
sigma = 0.5
new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                       final_cp=coordinates_final, radial_basis_function=radial_basis, **{'sigma_' : sigma})

plot_mesh(new_m, show=False, title='Our WS')
plt.scatter(coordinates_final[:,0], coordinates_final[:,1], color = 'r')
plt.show()

inds = np.where(coordinates_final[:,0] != 0)[0]

n_sim = 10
mesh_list, coord_list = [], []
plt.figure()
plt.title(name)

for sim in range(n_sim):
    start = True

    while not check_validity(new_m) or start:

        coordinates_final_new = copy.deepcopy(coordinates_final)
        start = False
        noise_x = np.random.normal(loc=0, scale=0.02, size=inds.shape[0])
        noise_y = np.random.normal(loc=0, scale=0.02, size=inds.shape[0])
        random_scaling = np.random.normal(loc=1.1, scale=0.02, size=1)
        random_rotation = np.random.uniform(low = -np.pi/8, high=np.pi/8)

        coordinates_final_new[inds, 0] += noise_x
        coordinates_final_new[inds, 1] += noise_y

        coordinates_final_new = stretch_cp(coordinates_final_new, random_scaling)
        coordinates_final_new = rotate_cp(coordinates_final_new, random_rotation)

        new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                           final_cp=coordinates_final_new, radial_basis_function=radial_basis, **{'sigma_': sigma})

        print(f'Trying mesh n : {sim}')

    mesh_list.append(new_m)
    _, _, scatter_border = extract_grid(new_m, flag_border=True)
    plt.scatter(scatter_border[:, 0], scatter_border[:, 1])

    save_name = f'{name}/{name}_{sim}'
    save_mesh(new_m, save_name, n_control_points)

plt.show()
