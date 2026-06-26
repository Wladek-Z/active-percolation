import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def angular_velocity():
    """
    Plot the angular velocity due to shear, due to flow velocity, and due 
    to the combined contribution of both.
    """
    # Define angular velocity formulas
    omega_shear = lambda yw, theta: 2 * (1 - 2 * yw) * np.cos(2 * theta)
    omega_flow = lambda yw: 2 * (2 * yw - 1)
    # Define horizontal coordinates of graph, angles to plot, and labels
    x = np.linspace(0, 1, 40)
    angles = np.arange(0, 5*np.pi/8, np.pi/8)
    labels = ["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"]
    # Calculate angular velocity due to flow vorticity
    O_f = omega_flow(x)

    # Define plotting function
    def plot(O_s, O, label):
        fig = plt.figure(figsize=[8, 6])
        plt.plot(x, O_f, color='blue', label="vorticity")
        plt.plot(x, O_s, color='red', linestyle='--', label="shear")
        plt.plot(x, O, color='black', linestyle='dotted', label="total")
        plt.xlabel("$y/w$")
        plt.ylabel(r"$\Omega$ [$\mathrm{Pe}_{\mathrm{f}}/w$]")
        plt.text(0.5, 0.95, r"$\theta$ = " + label, fontsize=16, ha='center', va='top', transform=plt.gca().transAxes)
        plt.legend(loc='lower center')
        plt.tight_layout()
        return fig
    
    # Loop over each angle to plot
    for i in range(len(angles)):
        O_s = omega_shear(x, angles[i])
        O = O_f + O_s
        plot(O_s, O, labels[i])
    
    plt.show()

if __name__ == "__main__":
    angular_velocity()
