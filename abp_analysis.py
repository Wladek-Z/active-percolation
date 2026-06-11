import argparse
import numpy as np
from matplotlib import pyplot as plt
from abp import ABP

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def power_phase_diagram_data(filename, N, T, dt, w, D, mu):
    """
    Collect data for a phase diagram of the ABP system by varying the two Peclet numbers
    and recording the power-law dependence of the resultant MSDs.

    Arguments:
        filename: file to record the data
        N: number of particle
        T: number of timesteps
        dt: timestep
        w: width of channel
        D: dimensionless diffusion constant
        mu: dimensionless repulsion strength
    """
    # Generate list of Peclet numbers to probe, from user input
    arr = input("Enter range of Peclet numbers to probe (Pe- Pe+ Pf- Pf+): ") 
    l = list(map(float,arr.split(' ')))
    Pe_list = np.linspace(l[0], l[1], 5)
    Pf_list = np.linspace(l[2], l[3], 5)

    with open(filename, 'w') as f:
        f.write("Pe,Pf,alpha\n")

        for Pe in Pe_list:
            for Pf in Pf_list:
                abp = ABP(N, T, dt, w, Pe, D, mu, Pf)
                r, e = abp.Run()
                _, msd = abp.get_MSD(r)
                a, _ = abp.get_MSD_fit(msd)
                # Track progress
                print(f"Pe = {Pe}, Pf = {Pf}, alpha = {a}")
                f.write(f"{Pe},{Pf},{a}\n")

def power_phase_diagram_plot(filename):
    """
    Plot the phase diagram for the ABP system in the Peclet number plane,
    displaying the fitted exponent of the late-time MSD.
    
    Arguments:
        filename: file to stored data
    """
    # Read in and interpret data
    Pe, Pf, a = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
    nPe, nPf = np.unique(Pe), np.unique(Pf)
    size_x = len(nPe)
    size_y = len(nPf)
    A = a.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nPe, nPf)
    # Plot heat map
    fig = plt.figure(figsize=[8, 6])
    plt.title("ABP Channel Phase Diagram")
    plt.pcolormesh(X, Y, A, cmap='viridis', shading='auto')
    plt.colorbar(label=r'MSD scaling exoponent, $\alpha$')
    plt.xlabel("active Peclet number, Pe")
    plt.ylabel("flow Peclet number, Pe$_f$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('--MSD', action='store_true', help='Find the mean square displacement')
    parser.add_argument('--trajectory', action='store_true', help='Find the particle trajectory')
    parser.add_argument('--FPTD', action='store_true', help='Obtain first-passage time distribution')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-Pe', type=float, default=5, help='Active Peclet number')
    parser.add_argument('-mu', type=float, default=10, help='Dimensionless force constant')
    parser.add_argument('-w',  type=float, default=10, help='Width of channel')
    parser.add_argument('-Pf', type=float, default=0, help='Flow Peclet number')
    parser.add_argument('-D', type=float, default=1, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-x', type=float, default=20, help='Distance along x-axis to check for first-passage')
    parser.add_argument('-f', type=str, default=None, help='Filepath to data for reading/writing')
    parser.add_argument('--PPD', action='store_true', help='Construct the power-law phase diagram')
    parser.add_argument('--data', action='store_true', help='Collect data for selected task')
    parser.add_argument('--plot', action='store_true', help='Plot results for selected task')
    args = parser.parse_args()

    if args.data:
        if args.PPD:
            power_phase_diagram_data(args.f, args.N, args.T, args.dt, args.w, args.D, args.mu)
    elif args.plot:
        if args.PPD:
            power_phase_diagram_plot(args.f)