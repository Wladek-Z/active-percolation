import argparse
import numpy as np
from matplotlib import pyplot as plt
from abp import ABP

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def phase_diagram(filename):
    """
    Plot the various phase diagrams of the ABP system in the Peclet
    number-persistence length plane.
    
    Arguments:
        filename: file to stored data
    """
    # Read in and interpret data
    lp_w, Pf_Ps, a, trap_frac, mean_vx = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
    nlp_w, nPf_Ps = np.unique(lp_w), np.unique(Pf_Ps)
    size_x = len(nlp_w)
    size_y = len(nPf_Ps)
    A = a.reshape(size_x, size_y).T
    TF = trap_frac.reshape(size_x, size_y).T
    VX = mean_vx.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nlp_w, nPf_Ps)

    def plot_pd(label, data):
        fig = plt.figure(figsize=[8, 6])
        plt.title("ABP Channel Phase Diagram")
        plt.pcolormesh(X, Y, data, cmap='viridis', shading='auto')
        plt.colorbar(label=label)
        plt.xlabel("$l_p/w$")
        plt.ylabel(r"Pe$_{\mathrm{f}}$/Pe$_{\mathrm{s}}$")
        plt.tight_layout()
        return fig
    
    plot_pd(r'MSD scaling exoponent, $\alpha$', A)
    plot_pd('trapping fraction', TF)
    plot_pd(r'mean longitudinal velocity [$\sigma D_r$]', VX)
    plt.show()

def pd_comparison(filename1, filename2):
    """
    Plot side-by-side phase diagram comparisons for two datasets.
    
    Arguments:
        filename1: filepath to dataset with shear
        filename2: filepath to dataset without shear
    """
    # Read in and interpret data (1)
    lp_w1, Pf_Ps1, a1, trap_frac1, mean_vx1 = np.loadtxt(filename1, delimiter=',', skiprows=1, unpack=True)
    nlp_w1, nPf_Ps1 = np.unique(lp_w1), np.unique(Pf_Ps1)
    size_x1 = len(nlp_w1)
    size_y1 = len(nPf_Ps1)
    A1 = a1.reshape(size_x1, size_y1).T
    TF1 = trap_frac1.reshape(size_x1, size_y1).T
    VX1 = mean_vx1.reshape(size_x1, size_y1).T
    X1, Y1 = np.meshgrid(nlp_w1, nPf_Ps1)

    # Read in and interpret data (2)
    lp_w2, Pf_Ps2, a2, trap_frac2, mean_vx2 = np.loadtxt(filename2, delimiter=',', skiprows=1, unpack=True)
    nlp_w2, nPf_Ps2 = np.unique(lp_w2), np.unique(Pf_Ps2)
    size_x2 = len(nlp_w2)
    size_y2 = len(nPf_Ps2)
    A2 = a2.reshape(size_x2, size_y2).T
    TF2 = trap_frac2.reshape(size_x2, size_y2).T
    VX2 = mean_vx2.reshape(size_x2, size_y2).T
    X2, Y2 = np.meshgrid(nlp_w2, nPf_Ps2)

    # Define function for plotting phase diagram comparison
    def plot_comparison_figure(label, data1, data2, cmap='viridis'):
        vmin = min(np.nanmin(data1), np.nanmin(data2))
        vmax = max(np.nanmax(data1), np.nanmax(data2))
        fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True)

        mesh1 = axes[0].pcolormesh(X1, Y1, data1, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Shear effects')
        axes[0].set_xlabel("$l_p/w$")
        axes[0].set_ylabel(r"Pe$_{\mathrm{f}}$/Pe$_{\mathrm{s}}$")

        mesh2 = axes[1].pcolormesh(X2, Y2, data2, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('No shear effects')
        axes[1].set_xlabel("$l_p/w$")
        axes[1].set_ylabel(r"Pe$_{\mathrm{f}}$/Pe$_{\mathrm{s}}$")

        fig.suptitle("ABP Channel Phase Diagram Comparison")
        fig.colorbar(mesh2, ax=axes, location='right', label=label)
        return fig

    plot_comparison_figure(r"MSD scaling exponent, $\alpha$", A1, A2)
    plot_comparison_figure("trapping fraction", TF1, TF2)
    plot_comparison_figure(r"mean longitudinal velocity [$\sigma D_r$]", VX1, VX2)
    plt.show()

def pdx_comparison(filename1):
    """
    Plot side-by-side phase diagram of total MSD scaling exponent compared
    to MSD scaling exponent in the longitudinal direction only.
    
    Arguments:
        filename1: filepath to dataset
    """
    # Read in and interpret data
    lp_w, Pf_Ps, a, a_x = np.loadtxt(filename1, delimiter=',', skiprows=1, unpack=True)
    nlp_w, nPf_Ps = np.unique(lp_w), np.unique(Pf_Ps)
    size_x = len(nlp_w)
    size_y = len(nPf_Ps)
    A = a.reshape(size_x, size_y).T
    A_x = a_x.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nlp_w, nPf_Ps)

    vmin = min(np.nanmin(A), np.nanmin(A_x))
    vmax = max(np.nanmax(A), np.nanmax(A_x))
    fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True)

    mesh1 = axes[0].pcolormesh(X, Y, A, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Total MSD')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel(r"Pe$_{\mathrm{f}}$/Pe$_{\mathrm{s}}$")

    mesh2 = axes[1].pcolormesh(X, Y, A_x, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Longitudinal MSD')
    axes[1].set_xlabel("$l_p/w$")
    axes[1].set_ylabel(r"Pe$_{\mathrm{f}}$/Pe$_{\mathrm{s}}$")

    fig.colorbar(mesh2, ax=axes, location='right', label=r"MSD scaling exponent, $\alpha$")
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default=None, help='Filepath to dataset for reading/writing')
    parser.add_argument('-f2', type=str, default=None, help='Filepath to second dataset, if applicable')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    parser.add_argument('--PD2', action='store_true', help='Compare the phase diagrams of systems with/without shear')
    parser.add_argument('--PDX', action='store_true', help='Compare the phase diagrams of alpha for total displacement and longitudinal displacement')
    args = parser.parse_args()

    if args.PD:
        phase_diagram(args.f)
    if args.PD2:
        pd_comparison(args.f, args.f2)
    if args.PDX:
        pdx_comparison(args.f)