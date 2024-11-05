import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(z):
    return z**3 - 1

def f_prime(z):
    return 3 * z**2

# Newton-Raphson solver
def solve_cnewton(z0, max_iter=200, tol=1e-9):
    z = z0
    for i in range(max_iter):
        z_next = z - f(z) / f_prime(z)
        if abs(z_next - z) < tol:
            return z_next, i, abs(f(z_next))
        z = z_next
    return 0, max_iter, abs(f(z))  # Indicate failure to converge

# Parameters
N = 400
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Grid in the complex plane
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y

# Initialize arrays to store results
roots = np.zeros(Z0.shape, dtype=complex)
iterations = np.zeros(Z0.shape, dtype=int)

# Solve for each point in the grid
for i in range(N):
    for j in range(N):
        z0 = Z0[i, j]
        root, iters, _ = solve_cnewton(z0)
        roots[i, j] = root
        iterations[i, j] = iters

# Plot the imaginary part of the root
plt.figure(figsize=(10, 8))
plt.imshow(np.angle(roots), extent=(x_min, x_max, y_min, y_max), cmap='hsv')
plt.colorbar(label='Phase of Root')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Root Convergence in Complex Plane (Phase)')
plt.show()

# Plot log of number of iterations
plt.figure(figsize=(10, 8))
plt.imshow(np.log10(iterations), extent=(x_min, x_max, y_min, y_max), cmap='inferno')
plt.colorbar(label='log10(Iterations)')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Iterations to Converge')
plt.show()
