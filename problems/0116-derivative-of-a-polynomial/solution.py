def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    # c: coefficient
    # n: exponent value
    # x: input
    return c * n * (x ** (n - 1))