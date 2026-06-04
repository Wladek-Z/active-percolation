import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import uniform_direction
import argparse

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def update(N, r, e, v, dt, d):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            r: particle positions
            e: particle orientations
            v: ABP velocity
            dt: timestep
            d: dimensionality of the system

        Returns:
            r_new: updated positions
            e_new: updated orientations
        """
        r_new = np.zeros_like(r)
        e_new = np.zeros_like(e)
        # Iterate over every particle
        for i in range(N):
            # Generate positional noise term
            r_noise = np.random.normal(0, 1, d)
            # Update position via forward difference scheme
            r_new[i] = r[i] + v * dt * e[i] + np.sqrt(2 * dt) * r_noise
            # Update orientation via dimensionally-dependent formula
            if d == 3:
                e_new[i] = orientation_3d(e[i], dt)
            else:
                e_new[i] = orientation_2d(e[i], dt)
        # Return updated position and orientation vectors
        return r_new, e_new

@njit
def orientation_3d(e, dt):
    """
    Calculate and return the new orientation vector in 3D using the rodrigues formula.
    
    Arguments:
        e: original orientation vector
        dt: timestep
    
    Returns:
        the updated orientation 
    """
    # Generate noise term
    noise = np.random.normal(0, 1, 3)
    # Calculate rotation vector
    rot_vec = np.sqrt(2 * dt) * noise
    # Calculate angle of rotation
    theta = np.linalg.norm(rot_vec)
    # Calculate axis of rotation
    k = rot_vec / theta
    # Return updated orientation vector
    return e * np.cos(theta) + np.cross(k, e) * np.sin(theta) + k * np.dot(k, e) * (1 - np.cos(theta))

@njit
def orientation_2d(e, dt):
    """
    Calculate and return the new orientation vector in 2D via rotation matrix update.
    
    Arguments:
        e: original orientation vector
        dt: timestep
        
    Returns:
        the updated orientation vector
    """
    # Calculate angle from x-axis of the original orientation vector
    theta = np.arctan2(e[1], e[0])
    # Generate noise term
    noise = np.random.normal(0, 1)
    # Calculate change in angle of orientation
    d_theta = np.sqrt(2 * dt) * noise
    # Calculate new angle
    new_theta = theta + d_theta
    # Return updated orientation vector
    return np.array([np.cos(new_theta), np.sin(new_theta)])

@njit
def run(N, r, e, v, dt, T, d):
    """
    Run the simulation for T timesteps and keep track of mean position squared.

    Arguments:
        N: number of particles
        r: particle positions
        e: particle orientations
        v: ABP velocity
        dt: timestep
        T: total number of timesteps
        d: dimensionality of the system

    Returns:
        H: history of mean position squared along each direction
    """
    # Initialise array to keep track of position data
    H = np.zeros((T, d))
    # Perform T iterations of the update procedure
    for i in range(T):
        # Update position and orientation
        r_old = r.copy()
        e_old = e.copy()
        r, e = update(N, r_old, e_old, v, dt, d)
        # Store mean squared displacement along each axis
        for j in range(d):
            msd = np.mean(r[:, j]**2)
            H[i, j] = msd
    return H
        

class ABP:
    """
    A single active Brownian particle submerged in a fluid with zero flow profile.
    """

    def __init__(self, N, dt, d):
        """
        Initialise N realisations of the same particle at the origin with random orientations.
        
        Arguments:
            N: number of realisations of the particle
            dt: timestep
            d: dimensionality of the system
        """
        self.N = N
        self.dt = dt
        self.dim = d
        # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
        distribution = uniform_direction(d)
        self.e = distribution.rvs(N)
        # Initialise positions
        self.r = np.zeros([N, d])


    def MSD(self, v, T):
        """
        Calculate the mean sqaure displacement of an ensemble of particles over time.
        Plot the results on a graph.
        
        Arguments:
            v: ABP velocity
            T: total number of timesteps
        """
        # Run simulation and retrieve mean squared position data
        xyz_squared = run(self.N, self.r, self.e, v, self.dt, T, self.dim)
        # Calculate mean square displacement
        msd = np.sum(xyz_squared, axis=1)

        # Create array of timesteps
        t = np.arange(1, T + 1) * self.dt
        # Calculate persistence time
        tau = 1 / (self.dim - 1)
        # Theoretical mean square displacement
        msd_theory = 2 * self.dim * t + 2 * v**2 * tau * t - 2 * v**2 * tau**2 * (1 - np.exp(-t / tau))

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Mean Square Displacement ({self.dim}D)")
        plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory')
        plt.xlabel(r"time [$1/D_r$]")
        plt.ylabel(r"$\langle r^2 \rangle$ [$D/D_r$]")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default='100', help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.01, help='Simulation timestep')
    parser.add_argument('-MSD', action='store_true', help='Calculate the mean square displacement')
    parser.add_argument('-T', type=int, default=10000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-v', type=float, default=1, help='ABP velocity')
    parser.add_argument('-d', choices=[2, 3], type=int, default=2, help='Dimensionality of the system')
    args = parser.parse_args()

    abp = ABP(args.N, args.dt, args.d)

    if args.MSD:
        abp.MSD(args.v, args.T)