import numpy as np
import matplotlib.pyplot as plt

# Define f(z) and f'(z)
def f(z):
    return 35 * z**9 - 180 * z**7 + 378 * z**5 - 420 * z**3 + 315 * z

def df(z):
    return 315 * z**8 - 1260 * z**6 + 1890 * z**4 - 1260 * z**2 + 315

# Newton-Raphson method
def newton_raphson(z0, tol=1e-6, max_iter=100):
    z = z0
    for i in range(max_iter):
        fz = f(z)
        dfz = df(z)
        if dfz == 0:  # Avoid division by zero
            return None
        z_new = z - fz / dfz
        if abs(z_new - z) < tol:  # Check for convergence
            return z_new
        z = z_new
    return None  # Did not converge within max_iter

# Example usage with an initial guess
initial_guess = 1 + 1j
root = newton_raphson(initial_guess)
print("Root found:", root)

# Define the plot grid
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y  # Initial complex guesses

# Prepare a result array for root convergence
roots = np.zeros(Z0.shape, dtype=complex)

# Find roots for each initial guess on the grid
for i in range(Z0.shape[0]):
    for j in range(Z0.shape[1]):
        roots[i, j] = newton_raphson(Z0[i, j])

# Plot the roots (basins of attraction)
plt.figure(figsize=(8, 8))
plt.imshow(np.angle(roots), extent=(-2, 2, -2, 2), cmap='hsv')
plt.colorbar(label="Root angle")
plt.title("Basins of Attraction for Newton-Raphson Method")
plt.xlabel("Real part of initial guess")
plt.ylabel("Imaginary part of initial guess")
plt.show()
