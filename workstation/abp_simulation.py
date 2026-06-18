import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abp import ABP

def phase_diagram_data(filename, N, T, dt, w, D, mu, l1, u1, l2, u2, n):
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
    """
    lp_w_list = np.round(np.linspace(l1, u1, n), 2)
    Pf_Pe_list = np.round(np.linspace(l2, u2, n), 2)

    with open(filename, 'w') as f:
        f.write("lp_w,Pf_Pe,alpha,back_frac,mean_vx\n")

        for lp_w in lp_w_list:
            Pe = lp_w * w
            for Pf_Pe in Pf_Pe_list:
                Pf = Pf_Pe * Pe
                abp = ABP(N, T, dt, w, Pe, D, mu, Pf)
                r, e = abp.Run()
                _, msd = abp.get_MSD(r)
                a, _ = abp.get_MSD_fit(msd)
                vx = abp.get_xspeed(r)
                back_frac = np.count_nonzero(vx < 0) / N / T
                mean_vx = np.mean(vx)
                # Track progress
                print(f"Pe = {Pe}, Pf = {Pf}, alpha = {a}, fraction = {back_frac}, mean x-velocity = {mean_vx}")
                f.write(f"{lp_w},{Pf_Pe},{a},{back_frac},{mean_vx}\n")

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
    parser.add_argument('-l1', type=float, help='Lower bound for data collection (first axis)')
    parser.add_argument('-u1', type=float, help='Upper bound for data collection (first axis)')
    parser.add_argument('-l2', type=float, help='Lower bound for data collection (second axis)')
    parser.add_argument('-u2', type=float, help='Upper bound for data collection (second axis)')
    parser.add_argument('-n', type=int, help='Number of data points')
    args = parser.parse_args()

    if args.PD:
        phase_diagram_data(args.f, args.N, args.T, args.dt, args.w, args.D, args.mu, args.l, args.u, args.n)