import argparse
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abp import ABP

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def phase_diagram_data(filename, N, T, dt, w, D, mu, l, u, n):
    """
    Collect data for a phase diagram of the ABP system by varying the two Peclet numbers
    and recording the power-law dependence of the resultant MSDs, as well as the fraction
    of time that the particles are travelling in the negative-x direction.

    Arguments:
        filename: file to record the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        w: width of channel
        D: dimensionless diffusion constant
        mu: dimensionless repulsion strength
        l: lower bound of data
        u: upper bound of data
        n: number of data points
    """
    number_list = np.round(np.linspace(l, u, n), 2)

    with open(filename, 'w') as f:
        f.write("Pe_w,Pf_Pe,alpha,back_frac\n")

        for Pe_w in number_list:
            Pe = Pe_w * w
            for Pf_Pe in number_list:
                Pf = Pf_Pe * Pe
                abp = ABP(N, T, dt, w, Pe, D, mu, Pf)
                r, e = abp.Run()
                _, msd = abp.get_MSD(r)
                a, _ = abp.get_MSD_fit(msd)
                vx = abp.get_xspeed(r)
                back_frac = np.count_nonzero(vx < 0) / N / T
                # Track progress
                print(f"Pe = {Pe}, Pf = {Pf}, alpha = {a}, fraction = {back_frac}")
                f.write(f"{Pe_w},{Pf_Pe},{a},{back_frac}\n")

def phase_diagram_plot(filename):
    """
    Plot the two phase diagrams for the ABP system in the Peclet number plane,
    displaying the fitted exponent of the late-time MSD on one, and the fraction 
    of time that particles have a negative instantaneous velocity in the x direction.
    
    Arguments:
        filename: file to stored data
    """
    # Read in and interpret data
    Pe_w, Pf_Pe, a, back_frac = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
    nPe_w, nPf_Pe = np.unique(Pe_w), np.unique(Pf_Pe)
    size_x = len(nPe_w)
    size_y = len(nPf_Pe)
    A = a.reshape(size_x, size_y).T
    BF = back_frac.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nPe_w, nPf_Pe)
    # Plot heat map 1
    fig = plt.figure(figsize=[8, 6])
    plt.title("ABP Channel Phase Diagram")
    plt.pcolormesh(X, Y, A, cmap='viridis', shading='auto')
    plt.colorbar(label=r'MSD scaling exoponent, $\alpha$')
    plt.xlabel("Pe/$w$")
    plt.ylabel("Pe$_f$/Pe")
    plt.tight_layout()
    plt.show()
    # Plot heat map 2
    fig = plt.figure(figsize=[8, 6])
    plt.title("ABP Channel Phase Diagram")
    plt.pcolormesh(X, Y, BF, cmap='viridis', shading='auto')
    plt.colorbar(label='fraction of timesteps with negative $x$-velocity')
    plt.xlabel("Pe/$w$")
    plt.ylabel("Pe$_f$/Pe")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-mu', type=float, default=10, help='Dimensionless force constant')
    parser.add_argument('-w',  type=float, default=10, help='Width of channel')
    parser.add_argument('-D', type=float, default=1, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-f', type=str, default=None, help='Filepath to data for reading/writing')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    parser.add_argument('--data', action='store_true', help='Collect data for selected task')
    parser.add_argument('--plot', action='store_true', help='Plot results for selected task')
    parser.add_argument('-l', type=float, help='Lower bound for data collection')
    parser.add_argument('-u', type=float, help='Upper bound for data collection')
    parser.add_argument('-n', type=int, help='Number of data points')
    args = parser.parse_args()
    
    if args.data:
        if args.PD:
            phase_diagram_data(args.f, args.N, args.T, args.dt, args.w, args.D, args.mu, args.l, args.u, args.n)
    elif args.plot:
        if args.PD:
            phase_diagram_plot(args.f)
