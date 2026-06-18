import argparse
import numpy as np
from matplotlib import pyplot as plt
from abp import ABP

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def phase_diagram_plot(filename):
    """
    Plot the various phase diagrams of the ABP system in the Peclet
    number-persistence length plane.
    
    Arguments:
        filename: file to stored data
    """
    # Read in and interpret data
    lp_w, Pf_Pe, a, back_frac, mean_vx = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
    nlp_w, nPf_Pe = np.unique(Pe_w), np.unique(Pf_Pe)
    size_x = len(nlp_w)
    size_y = len(nPf_Pe)
    A = a.reshape(size_x, size_y).T
    BF = back_frac.reshape(size_x, size_y).T
    VX = mean_vx.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nlp_w, nPf_Pe)
    # Plot heat map 1
    fig = plt.figure(figsize=[8, 6])
    plt.title("ABP Channel Phase Diagram")
    plt.pcolormesh(X, Y, A, cmap='viridis', shading='auto')
    plt.colorbar(label=r'MSD scaling exoponent, $\alpha$')
    plt.xlabel("$l_p/w$")
    plt.ylabel("Pe$_f$/Pe")
    plt.tight_layout()
    plt.show()
    # Plot heat map 2
    fig = plt.figure(figsize=[8, 6])
    plt.title("ABP Channel Phase Diagram")
    plt.pcolormesh(X, Y, BF, cmap='viridis', shading='auto')
    plt.colorbar(label='fraction of timesteps with negative $x$-velocity')
    plt.xlabel("$l_p/w$")
    plt.ylabel("Pe$_f$/Pe")
    plt.tight_layout()
    plt.show()
    # Plot heat map 3
    fig = plt.figure(figsize=[8, 6])
    plt.title("ABP Channel Phase Diagram")
    plt.pcolormesh(X, Y, VX, cmap='viridis', shading='auto')
    plt.colorbar(label=r'mean $x$-velocity [$\sigma D_r$]')
    plt.xlabel("$l_p/w$")
    plt.ylabel("Pe$_f$/Pe")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default=None, help='Filepath to data for reading/writing')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    args = parser.parse_args()

    if args.PD:
        phase_diagram_plot(args.f)