"""
This file contains code that I've written, but eventually decided not to use in my final simulation (e.g. code for 3D system, keyvalue class for argparser).
"""

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
    Run the simulation for T timesteps and keep track of positions and orientations.

    Arguments:
        N: number of particles
        r: particle positions
        e: particle orientations
        v: ABP velocity
        dt: timestep
        T: total number of timesteps
        d: dimensionality of the system

    Returns:
        pos_H: history of positions
        pos_O: history of orientations
    """
    # Initialise arrays to keep track of position/orientation data
    pos_H = np.zeros((T+1, N, 3))
    pos_O = np.zeros((T+1, N, 3))
    # Store position and orientation of each particle
    for j in range(N):
        for k in range(d):
            pos_H[0, j, k] = r[j, k]
            pos_O[0, j, k] = e[j, k]
    # Perform T iterations of the update procedure
    for i in range(1, T+1):
        # Update position and orientation
        r_old = r.copy()
        e_old = e.copy()
        r, e = update(N, r_old, e_old, v, dt, d)
        # Store position and orientation of each particle
        for j in range(N):
            for k in range(d):
                pos_H[i, j, k] = r[j, k]
                pos_O[i, j, k] = e[j, k]
    return pos_H, pos_O    


class ABP:
    """
    An ensemble of active Brownian particle submerged in a fluid with zero flow profile.
    Can be tested in either 2D or 3D.
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

    def Run(self, v, T, MSD, trajectory):
        """
        Run the simulation and display the desired results.
        
        Arguments:
            v: ABP velocity
            T: total number of timesteps
            MSD: display mean square displacement
            trajectory: display trajectory
        """
        # Run simulation and retrieve data
        pos, orient = run(self.N, self.r, self.e, v, self.dt, T, self.dim)
        # Calculate and display mean square displacement
        if MSD:
            self.MSD(pos, v, T)
        # Calculate and display trajectories
        if trajectory and (self.dim == 2):
            self.Trajectory(pos)


    def MSD(self, data, v, T):
        """
        Calculate the mean sqaure displacement of an ensemble of particles over time.
        Plot the results on a graph.

        Arguments:
            data: ABP position history
            v: ABP velocity
            T: total number of timesteps
        """
        # Calculate mean square displacement along each Cartesian direction
        msds = np.mean((data - data[0])**2, axis=1)
        # Calculate total mean square displacement, omitting initial position for logscale
        msd = np.sum(msds, axis=1)[1:]

        # Create array of timesteps
        t = np.arange(1, T + 1) * self.dt
        # Calculate persistence time
        tau = 1 / (self.dim - 1)
        # Theoretical mean square displacement
        msd_theory = 2 * self.dim * t + 2 * v**2 * tau * t - 2 * v**2 * tau**2 * (1 - np.exp(-t / tau))
        # Theoretical msd for ballistic and diffusive regimes
        msd_b = v**2 * t**2 + 2 * self.dim * t
        msd_d = (2 * self.dim + 2 * v**2 * tau) * t 

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Mean Square Displacement")
        plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive')
        plt.axvline(tau, color='orange', label=r'$\tau_r$')
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
            data: ABP position history
        """
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Particle Trajectory")
        
        for i in range(self.N):
            x, y = data[:, i, 0], data[:, i, 1]
            plt.plot(x, y, color='black')
        
        plt.xlabel(r"$x$ position [$\sigma$]")
        plt.ylabel(r"$y$ position [$\sigma$]")
        plt.tight_layout()
        plt.show()

    def PDF(self, p, o):
        """
        Obtain the probability density functions in both positional and
        orientational space.

        Arguments:
            p: position data
            o: orientation data
        """
        # Isolate vertical position from position data
        y = p[:, :, 1].flatten() / self.width

        # Construct positional PDF
        pdf1, edges1 = np.histogram(y, bins=100, density=True)
        x1 = 0.5 * (edges1[:-1] + edges1[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x1, pdf1, color='black')
        plt.title(f"Spatial distribution of ABPs: " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel("height along channel, $y/w$")
        plt.ylabel("probability density, $P(y/w)$")
        plt.xlim(0, 1)
        plt.tight_layout()

        # Isolate orientation angles from orientation data
        theta = np.arctan2(o[:, :, 1].flatten(), o[:, :, 0].flatten())

        # Construct orientational PDF
        pdf2, edges2 = np.histogram(theta, bins=100, density=True)
        x2 = 0.5 * (edges2[:-1] + edges2[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x2, pdf2, color='black')
        plt.title("Orientational distribution of ABPs: " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel(r"orientation angle, $\theta$")
        plt.ylabel(r"probability density, $P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Calculate orientation angles in the bulk and near the surface
        theta_blk_o, theta_sfc_o = self.blk_sfc_orient(p, theta)
        # Construct PDF showing the distribution of swimming direction in the bulk
        pdf3, edges3 = np.histogram(theta_blk_o, bins=100, density=True)
        x3 = 0.5 * (edges3[:-1] + edges3[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x3, pdf3, color='black')
        plt.title("Orientational distribution (bulk): " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel(r"orientation angle, $\theta$")
        plt.ylabel(r"probability density, $P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Construct PDF showing the distribution of swimming direction in the bulk
        pdf4, edges4 = np.histogram(theta_sfc_o, bins=100, density=True)
        x4 = 0.5 * (edges4[:-1] + edges4[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x4, pdf4, color='black')
        plt.title("Orientational distribution (surface): " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel(r"orientation angle, $\theta$")
        plt.ylabel(r"probability density, $P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Calculate angles of velocity vectors
        vx = self.get_xspeed(p)
        vy = self.get_yspeed(p)
        theta_v = np.arctan2(vy.flatten(), vx.flatten())
        # Construct PDF showing the distribution of swimming direction
        pdf5, edges5 = np.histogram(theta_v, bins=100, density=True)
        x5 = 0.5 * (edges5[:-1] + edges5[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x5, pdf5, color='black')
        plt.title("Swimming direction distribution: " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel(r"swimming direction, $\theta_{\mathrm{s}}$")
        plt.ylabel(r"probability density, $P(\theta_{\mathrm{s}})$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Calculate swimming angles in the bulk and near the surface
        theta_blk_s, theta_sfc_s = self.blk_sfc_swim(p, theta_v)
        # Construct PDF showing the distribution of swimming direction in the bulk
        pdf6, edges6 = np.histogram(theta_blk_s, bins=100, density=True)
        x6 = 0.5 * (edges6[:-1] + edges6[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x6, pdf6, color='black')
        plt.title("Swimming direction distribution (bulk): " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel(r"swimming direction, $\theta_{\mathrm{s}}$")
        plt.ylabel(r"probability density, $P(\theta_{\mathrm{s}})$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Construct PDF showing the distribution of swimming direction in the bulk
        pdf7, edges7 = np.histogram(theta_sfc_s, bins=100, density=True)
        x7 = 0.5 * (edges7[:-1] + edges7[1:])

        fig = plt.figure(figsize=[8, 6])
        plt.plot(x7, pdf7, color='black')
        plt.title("Swimming direction distribution (surface): " + r"Pe$_{\mathrm{s}}$ = " + f"{self.Ps}, " + r"Pe$_{\mathrm{f}}$ = " + f"{self.Pf}")
        plt.xlabel(r"swimming direction, $\theta_{\mathrm{s}}$")
        plt.ylabel(r"probability density, $P(\theta_{\mathrm{s}})$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        plt.show()


# Create a key value class
class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
        
        for value in values:
            # Split it into key and value
            key, value = value.split('=')
            if key == 'wall':
                value = float(value)
            # Assign into dictionary
            getattr(namespace, self.dest)[key] = value



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.01, help='Simulation timestep')
    parser.add_argument('-MSD', action='store_true', help='Find the mean square displacement')
    parser.add_argument('-trajectory', action='store_true', help='Find the particle trajectory')
    parser.add_argument('-t', type=int, default=10000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-v', type=float, default=1, help='ABP velocity')
    parser.add_argument('-d', choices=[2, 3], type=int, default=2, help='Dimensionality of the system')
    args = parser.parse_args()

    x = np.linspace(0, 4, 9)
    print(x)