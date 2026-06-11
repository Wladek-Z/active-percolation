import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import uniform_direction
import argparse
import random

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def run(N, r, e, T, dt, w, Pe, D, mu, Pf):
    """
    Run the simulation for T timesteps, recording all positions and orientations.

    Arguments:
        N: number of particles
        r: particle positions
        e: particle orientations
        T: total number of timesteps
        dt: timestep
        w: width of the channel
        Pe: active Peclet number
        D: diffusion number
        mu: repulsion number
        Pf: flow Peclet number

    Returns:
        pos_H: history of positions
        pos_O: history of orientations
    """
    # Initialise arrays to keep track of position/orientation data
    pos_H = np.zeros((T+1, N, 2))
    pos_O = np.zeros((T+1, N, 2))
    # Store position and orientation of each particle
    for j in range(N):
        for k in range(2):
            pos_H[0, j, k] = r[j, k]
            pos_O[0, j, k] = e[j, k]
    # Perform T iterations of the update procedure
    for i in range(1, T+1):
        # Update position and orientation
        r_old = r.copy()
        e_old = e.copy()
        r, e = update(N, r_old, e_old, dt, w, Pe, D, mu, Pf)
        # Store position and orientation of each particle
        for j in range(N):
            for k in range(2):
                pos_H[i, j, k] = r[j, k]
                pos_O[i, j, k] = e[j, k]
    return pos_H, pos_O   

@njit
def update(N, r, e, dt, w, Pe, D, mu, Pf):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            r: particle positions
            e: particle orientations
            dt: timestep
            w: width of the channel
            Pe: active Peclet number
            D: diffusion number
            mu: repulsion number
            Pf: flow Peclet number

        Returns:
            r_new: updated positions
            e_new: updated orientations
        """
        r_new = np.zeros_like(r)
        e_new = np.zeros_like(e)
        # Initialise critical distance to obstacles
        d_crit = 2**(1/6)
        # Iterate over every particle
        for i in range(N):
            # Compute active velocity term
            r_active = dt * Pe * e[i] 
            # Generate positional noise term
            r_noise = np.sqrt(2 * dt) * D * np.random.normal(0, 1, 2) 
            # Update position via forward difference scheme
            r_new[i] = r[i] + r_active + r_noise
            # Incorporate correction due to fluid flow
            r_new[i, 0] += dt * Pf * 4 * r[i, 1] / w * (1 - r[i, 1] / w) 
            # Initialise correction due to repulsive interactions with obstacles
            correction = np.zeros(2)
            # Check for obstacles
            if r[i, 0] < d_crit:
                # Add repulsive interaction due to left wall
                sep = np.array([r[i, 0], 0])
                correction += repulsion(sep)
            if r[i, 1] < d_crit:
                # Add repulsive interaction due to lower wall
                sep = np.array([0, r[i, 1]])
                correction += repulsion(sep)
            if (w - r[i, 1]) < d_crit:
                # Add repulsive interaction due to upper wall
                sep = np.array([0, (r[i, 1] - w)])
                correction += repulsion(sep)
            r_new[i] += dt * mu * correction
            # Update orientation
            e_new[i] = orientation(e[i], dt, w, Pf, r[i, 1])
        # Return updated position and orientation vectors
        return r_new, e_new

@njit
def repulsion(sep):
    """
    Calculate correction due to repulsive interactions with obstactle.
    
    Arguments:
        sep: separation vector from particle to obstacle
    
    Returns:
        repulsive force acting on particle
    """
    r = np.linalg.norm(sep)
    return 24 * (2 / r**14 - 1 / r**8) * sep

@njit
def orientation(e, dt, w, Pf, y):
    """
    Calculate and return the new orientation vector in 2D via rotation matrix update.
    
    Arguments:
        e: original orientation vector
        dt: timestep
        w: width of channel
        Pf: flow Peclet number
        y: height of particle within channel
        
    Returns:
        the updated orientation vector
    """
    # Calculate change in orientation due to noise
    d_theta_noise = np.sqrt(2 * dt) * np.random.normal(0, 1) 
    # Calculate change in orientation due to angular velocity from flow profile
    d_theta_flow = dt * Pf * 2 / w * (2 * y / w - 1) 
    # Calculate angle from x-axis of the original orientation vector
    theta = np.arctan2(e[1], e[0])
    # Calculate new angle
    new_theta = theta + d_theta_noise + d_theta_flow
    # Return updated orientation vector
    return np.array([np.cos(new_theta), np.sin(new_theta)])

@njit
def positions(N, width):
    """
    Generate the positions of N particles near the beginning of a channel.
    
    Arguments:
        N: number of particles
        width: width of channel
    
    Returns:
        r: positions of each particle
    """
    x_min = 2**(1/6)
    x_max = 0.5 * width
    y_min = 2**(1/6)
    y_max = width - 2**(1/6)
    # Initialise positions
    r = np.zeros((N, 2))
    # Randomly choose x and y components within specified range
    for i in range(N):
        r[i, 0] = random.uniform(x_min, x_max)
        r[i, 1] = random.uniform(y_min, y_max)
    # Return position array
    return r


class ABP:
    """
    An ensemble of active Brownian particles submerged in fluid, for a given system geometry (2D).
    """

    def __init__(self, N, T, dt, width, Pe, D, mu, Pf):
        """
        Initialise N realisations of the same particle at the origin with random orientations.
        
        Arguments:
            N: number of realisations of the particle
            T: total number of timesteps
            dt: timestep
            width: width of channel
            Pe: active Peclet number
            D: diffusion number
            mu: repulsion number
            Pf: flow Peclet number
            
        """
        # Initialise variables
        self.N = N
        self.T = T
        self.dt = dt
        self.width = width
        self.Pe = Pe
        self.D = D
        self.mu = mu
        self.Pf = Pf
        # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
        distribution = uniform_direction(2)
        self.e = distribution.rvs(N)
        # Initialise positions
        self.r = positions(N, width)
        
 
    def Run(self):
        """
        Run the simulation and return the results.
        
        Returns:
            pos: position history of each particle
            orient: orientation history of each particle
        """
        # Run simulation and retrieve data
        pos, orient = run(self.N, self.r, self.e, self.T, self.dt, self.width, self.Pe, self.D, self.mu, self.Pf)
        return pos, orient        


    def get_MSD(self, data):
        """
        Calculate and return the mean square displacement.
        
        Arguments:
            data: position history
        
        Returns:
            msd_xyz: the MSD along each Cartesian direction
            msd_tot: the total mean square displacement
        """
        # Calculate mean square displacement along each Cartesian direction
        msd_xyz = np.mean((data - data[0])**2, axis=1)
        # Calculate total mean square displacement
        msd_tot = np.sum(msd_xyz, axis=1)
        # Delete first data point to avoid conflict with logscale
        return np.delete(msd_xyz, 0), np.delete(msd_tot, 0)
    
    def get_MSD_fit(self, msd):
        """
        Obtain the power-law dependence of the late-time mean square displacement.
        
        Argument:
            msd: mean square displacement data
        
        Returns:
            a: fitted power
            b: fitted logarithm of prefactor
        """
        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # System is 2-dimensional
        d = 2
        # Calculate persistence time
        tau = 1 / (d - 1) 
        # Consider only late-time data, i.e. >> tau
        late = t >= 10 * tau
        y = np.log(msd[late])
        x = np.log(t[late])
        # Fit to a 1st degree polynomial
        a, b = np.polyfit(x, y, 1)
        # return fitted parameters
        return a, b

    def MSD(self, data):
        """
        Calculate the mean sqaure displacement of an ensemble of particles over time.
        Plot the results on a graph.

        Arguments:
            data: position history
        """
        _, msd = self.get_MSD(data)

        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # System is 2-dimensional
        d = 2
        # Calculate persistence time
        tau = 1 / (d - 1) 
        # Theoretical mean square displacement
        msd_theory = 2 * d * self.D**2 * t + 2 * self.Pe**2 * tau * t - 2 * self.Pe**2 * tau**2 * (1 - np.exp(-t / tau))
        # Theoretical msd for ballistic and diffusive regimes
        msd_b = self.Pe**2 * t**2 + 2 * d * self.D**2 * t
        msd_d = 2 * t * (d * self.D**2 + self.Pe**2 * tau)
        # Perform fit to late-time data
        a, b = self.get_MSD_fit(msd)
        t_fit = np.linspace(tau/self.dt, self.T, 100) * self.dt
        msd_fit = np.exp(b) * t_fit**a

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Mean Square Displacement: Pe = {self.Pe}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_fit, color='magenta', label=r'$\alpha$ = ' + f'{np.round(a, 2)}')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$\tau_r$')
        plt.xlabel(r"time [$1/D_r$]")
        plt.ylabel(r"$\langle r^2 \rangle$ [$\sigma^2$]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def Trajectory(self, data):
        """
        Retrieve the positions of an ensemble of particles over time.
        Plot results on a graph.
        
        Arguments:
            data: position history
        """
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Particle Trajectory: Pe = {self.Pe}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        
        for i in range(self.N):
            x, y = data[:, i, 0], data[:, i, 1]
            start_x, start_y = data[0, i, 0], data[0, i, 1]
            plt.scatter(start_x, start_y, color='lime', s=20, zorder=1)
            end_x, end_y = data[-1, i, 0], data[-1, i, 1]
            plt.scatter(end_x, end_y, color='red', s=20, zorder=1)
            plt.plot(x, y, color='black', zorder=-1)

        plt.xlabel(r"$x$ position [$\sigma$]")
        plt.ylabel(r"$y$ position [$\sigma$]")
        plt.xlim(0)
        plt.ylim(0, self.width)
        plt.tight_layout()
        plt.show()
    
    def FPTD(self, data, target):
        """
        Obtain the first-passage time distribution for an ensemble of particles
        to reach a point x along the x-axis.
        Display results as a histogram.
        
        Arguments:
            data: ABP position history
            target: distance to check for first passage
        """
        # Get all x positions
        x = data[:, :, 0]
        # Define condition for particle crossing the target
        crossed = x >= target
        # Identify all particles that successfully crossed the target
        has_crossed = np.any(crossed, axis=0)
        # Obtain first crossing of each particle
        first = np.argmax(crossed, axis=0)
        # Discard all unsuccessful attempts and convert to time units
        fpt = first[has_crossed] * self.dt
        success_rate = len(fpt) / self.N * 100

        # Build histogram of the FPTD
        fig = plt.figure(figsize=[8, 6])
        plt.hist(fpt, bins='auto')
        plt.title(f"First-Passage Time Distribution\nTarget: {target}" + r"$\sigma$, " + f"Success Rate: {success_rate}%")
        plt.xlabel("time [$1/D_r$]")
        plt.ylabel("crossings")
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
    args = parser.parse_args()

    abp = ABP(args.N, args.T, args.dt, args.w, args.Pe, args.D, args.mu, args.Pf)
    pos, orient = abp.Run()

    if args.MSD:
        abp.MSD(pos)
    if args.trajectory:
        abp.Trajectory(pos)
    if args.FPTD:
        abp.FPTD(pos, args.x)