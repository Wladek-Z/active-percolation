import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abp import ABP

def phase_diagram(filename, N, T, dt, w, D, mu, l1, u1, l2, u2, n, G):
    """
    Collect data for various phase diagrams of the ABP system by varying the ration of both
    Peclet numbers and the ratio of the persistence length and the channel width.

    Arguments:
        filename: file to record the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        w: width of channel
        D: dimensionless diffusion constant
        mu: dimensionless repulsion strength
        l1: lower bound of data along first axis
        u1: upper bound of data along first axis
        l2: lower bound of data along second axis
        u2: upper bound of data along second axis
        n: number of data points
        G: elongation factor
    """
    lp_w_list = np.round(np.linspace(l1, u1, n), 3)
    Pf_Ps_list = np.round(np.linspace(l2, u2, n), 3)

    with open(filename, 'w') as f:
        f.write("lp_w,Pf_Ps,alpha,trap_frac,mean_vx\n")

        for lp_w in lp_w_list:
            Ps = lp_w * w
            for Pf_Ps in Pf_Ps_list:
                Pf = Pf_Ps * Ps
                abp = ABP(N, T, dt, w, Ps, D, mu, Pf, G)
                r_hist, e_hist = abp.Run()
                # Decorrelate the position and orientation data
                r = r_hist[::abp.step]
                e = e_hist[::abp.step]
                # Get results
                _, msd = abp.get_MSD(r_hist)
                a, _ = abp.get_MSD_fit(msd)
                vx = abp.get_xspeed(r)
                trap = abp.trapping_index(r, vx)
                trap_frac = np.count_nonzero(trap) / (trap.shape[0] * trap.shape[1])
                mean_vx = np.mean(vx)
                # Track progress
                print(f"Ps = {Ps}, Pf = {Pf}, alpha = {a}, trapping fraction = {trap_frac}, mean x-velocity = {mean_vx}")
                f.write(f"{lp_w},{Pf_Ps},{a},{trap_frac},{mean_vx}\n")

def phase_diagram_x(filename, N, T, dt, w, D, mu, l1, u1, l2, u2, n, G):
    """
    Collect data for various phase diagrams of the ABP system by varying the ration of both
    Peclet numbers and the ratio of the persistence length and the channel width.

    Arguments:
        filename: file to record the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        w: width of channel
        D: dimensionless diffusion constant
        mu: dimensionless repulsion strength
        l1: lower bound of data along first axis
        u1: upper bound of data along first axis
        l2: lower bound of data along second axis
        u2: upper bound of data along second axis
        n: number of data points
        G: elongation factor
    """
    lp_w_list = np.round(np.linspace(l1, u1, n), 6)
    Pf_Ps_list = np.round(np.linspace(l2, u2, n), 6)

    with open(filename, 'w') as f:
        f.write("lp_w,Pf_Ps,alpha,alpha_x\n")

        for lp_w in lp_w_list:
            Ps = lp_w * w
            for Pf_Ps in Pf_Ps_list:
                Pf = Pf_Ps * Ps
                abp = ABP(N, T, dt, w, Ps, D, mu, Pf, G)
                r, e = abp.Run()
                msd_xy, msd = abp.get_MSD(r)
                a, b = abp.get_MSD_fit(msd)
                msd_x = msd_xy[:, 0]
                a_x, b_x = abp.get_MSD_fit(msd_x)
                # Track progress
                print(f"Ps = {Ps}, Pf = {Pf}, alpha = {a}, alphaX = {a_x}")
                f.write(f"{lp_w},{Pf_Ps},{a},{a_x}\n")

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
    parser.add_argument('--PDX', action='store_true', help='Record MSD scaling exponents, including x-direction')
    parser.add_argument('-l1', type=float, help='Lower bound for data collection (first axis)')
    parser.add_argument('-u1', type=float, help='Upper bound for data collection (first axis)')
    parser.add_argument('-l2', type=float, help='Lower bound for data collection (second axis)')
    parser.add_argument('-u2', type=float, help='Upper bound for data collection (second axis)')
    parser.add_argument('-n', type=int, help='Number of data points')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    parser.add_argument('--MSD', action='store_true', help='Find the mean square displacement')
    parser.add_argument('--trajectory', action='store_true', help='Find the particle trajectory')
    args = parser.parse_args()

    if args.PD:
        phase_diagram(args.f, args.N, args.T, args.dt, args.w, args.D, args.mu, args.l1, args.u1, args.l2, args.u2, args.n, args.G)
    elif args.PDX:
        phase_diagram_x(args.f, args.N, args.T, args.dt, args.w, args.D, args.mu, args.l1, args.u1, args.l2, args.u2, args.n, args.G)
