import numpy as np

# Define the function and its derivative
def f(z):
    return z**3 - 1

def f_prime(z):
    return 3 * z**2

# Newton-Raphson solver for complex numbers
def solve_cnewton(z0, max_iter=200, tol=1e-9):
    z = z0
    for i in range(max_iter):
        z_next = z - f(z) / f_prime(z)
        if abs(z_next - z) < tol:
            return z_next, i + 1, abs(f(z_next))
        z = z_next
    return 0, max_iter, abs(f(z))  # Indicates no convergence

# Parameters
N = 400               # Grid size
x_min, x_max = -2, 2  # Real axis range
y_min, y_max = -2, 2  # Imaginary axis range

# Generate grid in the complex plane
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y  # Initial guesses in the complex plane

# Prepare to store results
results = []

# Apply Newton-Raphson to each point in the grid
for i in range(N):
    row_results = []
    for j in range(N):
        z0 = Z0[i, j]
        root, iterations, residual = solve_cnewton(z0)
        x0, y0 = z0.real, z0.imag
        k_z = root if residual < 1e-9 else 0  # Indicate failure with 0
        row_results.append(f"{x0:.4f} {y0:.4f} {k_z.real:.4f} {k_z.imag:.4f} {residual:.4e} {np.log10(iterations):.4f}")
    results.append(" ".join(row_results))

# Print or save the results
with open("output.txt", "w") as file:
    for row in results:
        file.write(row + "\n\n")  # Add blank line after each row for readability
    file.close()






    

    
