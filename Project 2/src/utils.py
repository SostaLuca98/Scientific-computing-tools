import shutil
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.spatial.distance import cdist

import RBF


def choose_rbf(name=None):
    """

    :param name: name of radial basis function, either None, l2, l1, inv, mixed
    :return: radial basis function callable
    """
    if name is None or name == 'l2': return RBF.radial_basis
    assert (name in ['inv', 'l1', 'mixed']), 'There is no rbf function with this name'
    if name == 'inv': return RBF.radial_basis_inv
    elif name == 'l1' : return RBF.radial_basis1
    elif name == 'mixed' : return RBF.radial_basisM

def extract_grid(msh, flag_border=False):

    """
    :param msh: object of type mesh
    :param flag_border: True if indices of border points have to be returned
    :return: points of the mesh, cells of connectivity, optionally indices of border points
    """
    points = msh.points[:, 0:2]
    cells = msh.cells_dict['triangle']
    border_points = points[msh.cells[0].data[:, 0]]

    return (points, cells, border_points) if flag_border else (points, cells)

def plot_mesh(msh, title=None, show=True):

    """
    :param msh: the mesh to be plotted
    :param title: optional title to be shown
    :param show:: whether to plot the graph or store for subplots
    :return: nothing
    """

    points, cells = extract_grid(msh)

    fontsize = 15

    if show:
        plt.figure(figsize=(20, 12), dpi=32)
        fontsize = 45

    if title is not None:
        plt.title(title, fontsize=fontsize)

    triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
    plt.triplot(triangulation, '--')
    #plt.xlim(0, 2.5)
    #plt.ylim(0, 1.5)

    if show:
        plt.show()

    return

def transform_mesh(msh, radial_basis_function, starting_cp, final_cp,
                   flag_final_control_points = True, **kwargs):
    """

    :param msh: mesh template to be modified
    :param radial_basis_function: function to interpolate points
    :param starting_cp: starting position of all control points
    :param flag_final_control_points: True if final_cp is final position of the control points, False if
            final_cp is the displacement
    :param final_cp: Final position of control points or displacement
    :param kwargs: sigma or generic arguments for radial basis function
    :return: new mesh
    """
    msh_template = deepcopy(msh)

    assert (starting_cp.shape[0] == final_cp.shape[0]), 'Check shapes of control points'
    displacement = final_cp - starting_cp if flag_final_control_points else final_cp
    S = cdist(starting_cp, starting_cp, metric=radial_basis_function, **kwargs)

    points, cells = extract_grid(msh_template)
    W = np.linalg.solve(S, displacement)

    phi_x = points[:, 0]
    phi_y = points[:, 1]

    kernel_matrix = cdist(points, starting_cp, metric=radial_basis_function, **kwargs)
    phi_x = phi_x + np.dot(W[:, 0], kernel_matrix.T)
    phi_y = phi_y + np.dot(W[:, 1], kernel_matrix.T)

    points = np.array([phi_x, phi_y]).T
    msh_template.points[:,0:2] = points

    return msh_template

def save_mesh(msh, name, n_control_points):
    """

    :param msh: mesh to save
    :param name: name of the shape
    :param n_control_points: number of control points used
    :return: nothing
    """
    shutil.copy("save_template.msh", f'{name}_{n_control_points}.msh')

    lines_to_append = []
    for ii in range(msh.points.shape[0]):
        lines_to_append.append(
            "%d %.16f %.16f %.16f \n" % (ii + 1, msh.points[ii][0], msh.points[ii][1], msh.points[ii][2]))

    input_file = open(f'{name}_{n_control_points}.msh', 'r').readlines()
    write_file = open(f'{name}_{n_control_points}.msh', 'w')

    inserted = False
    for line in input_file:
        write_file.write(line)
        if "5609\n" in line and not inserted:
            inserted = True
            for item in lines_to_append:
                new_line = item
                write_file.write(new_line)
    write_file.close()

    print('Saving completed')

    return

def check_validity(msh):

    points, cells = extract_grid(msh)
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
    try:
        tri.TrapezoidMapTriFinder(triangulation)
        return True

    except RuntimeError:
        return False

def stretch_cp(final_control_points, coeff=1):

    inds = np.where(final_control_points[:, 0] != 0)[0]
    new_control_points = deepcopy(final_control_points)
    new_control_points[inds,0] = coeff * final_control_points[inds,0]

    return new_control_points
def inflate_cp(final_control_points, coeff=1):

    inds = np.where(final_control_points[:, 0] != 0)[0]
    new_control_points = deepcopy(final_control_points)
    new_control_points[inds,:] = coeff * final_control_points[inds,:]

    return new_control_points

def rotate_cp(final_control_points, theta=0):
    """

    :param final_control_points: final control points
    :param theta: between -pi/12 and +pi/12
    :return:
    """
    inds = np.where(final_control_points[:, 0] != 0)[0]
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    new_control_points = deepcopy(final_control_points)
    new_control_points[inds, :] = np.dot(new_control_points[inds,:], rotation_matrix.T)

    return new_control_points
