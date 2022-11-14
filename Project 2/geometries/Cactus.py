import matplotlib.pyplot as plt
import meshio
import numpy as np
import sys
sys.path.append("../scripts")
from utils import plot_mesh, extract_grid, transform_mesh, choose_rbf, save_mesh, check_validity, stretch_cp

msh_template = meshio.read('LAA_template.msh')
template_points, template_cells, template_border = extract_grid(msh_template, flag_border=True)
radial_basis = choose_rbf(name='l2')

# control points
#p0 = np.logical_and(template_points[:,0] == 0, template_points[:,1] == 0)
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
                             [1.1, 0.85],
                             [1.75, -0.05],
                             [1.72, -0.3],
                             [0.8, 0.15],
                             [1.9, -0.25],
                             [0.4, 0.175],
                             [1.15, 0.035],
                             [1.45, -0.2],
                             [1.7, 0.47],   #1.9, 0.27],
                             [1.48, 0.82],
                             [0.47, 1.10],
                             [0.2, 1],
                             [0.95, 1.2],
                             [1.75, 0.35],
                             [1.65, 0.68],
                             [1.62, 0.15]
                             ])

coordinates_final = np.concatenate([xy, coordinates_final])

#parameter
sigma = 0.5
new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                       final_cp=coordinates_final, radial_basis_function=radial_basis, **{'sigma_' : sigma})

inds = np.where(coordinates_final[:,0] != 0)[0]
counter = 0
while not check_validity(new_m) and False :

    noise_x = np.random.normal(loc=0, scale=0.1, size=inds.shape[0])
    noise_y = np.random.normal(loc=0, scale=0.1, size=inds.shape[0])
    #random_scaling = np.random.normal(loc=1.1, scale=0.1, size=1)

    coordinates_final[inds, 0] += noise_x
    coordinates_final[inds, 1] += noise_y

    #coordinates_final = stretch_cp(coordinates_final, random_scaling)

    new_m = transform_mesh(msh_template, starting_cp=coordinates_init, flag_final_control_points=True,
                           final_cp=coordinates_final, radial_basis_function=radial_basis, **{'sigma_': sigma})
    counter = counter + 1
    print(f'Trying mesh n : {counter}')

windsock_points, windsock_cells, windsock_border = extract_grid(new_m, flag_border=True)

x_min, x_max = 0, 2.8
y_min, y_max = -0.75, 1.5

# plt.subplot(1,2,1)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plot_mesh(msh_chicken, title='Chicken Wing', show=False)

# plt.subplot(1,2,2)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.figure()
plot_mesh(new_m, show=False)
plt.scatter(coordinates_final[:,0], coordinates_final[:,1], color = 'r')
plt.show()

# Saving procedure
name = 'CACTUS'
#if check_validity(new_m):
#    save_mesh(new_m, name, n_control_points)
