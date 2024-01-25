def derive(f, x, h=0.0001):
    """
    Calculate the numerical derivative of the function f at point x.
    """
    derivative_approximation = (f(x + h) - f(x)) / h
    return derivative_approximation

# Example usage:
# Define your function f(x)
def f(x):
    return x**2

# Choose a point x
x_value = 2.0

# Calculate the numerical derivative at the chosen point
derivative_at_x = derive(f, x_value)

print(f"The derivative of f(x) at x = {x_value} is approximately {derivative_at_x}")
