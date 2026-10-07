import numpy as np

def poly_derivative(coeffs: list, x: float) -> float:
    evaluation : int = 0

    for power, coeff in enumerate(reversed(coeffs[:-1]), start=1):
        evaluation += coeff * power * (x ** (power - 1))

    return evaluation

def poly(coeffs: list, x: float) -> float:
    evaluation : int = 0

    for power, coeff in enumerate(reversed(coeffs)):
        evaluation += coeff * (x ** power)
    
    return evaluation

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """

    h_x     : float = poly(h_coeffs, x)
    h_der_x : float = poly_derivative(h_coeffs, x)

    g_x     : float = poly(g_coeffs, x)
    g_der_x : float = poly_derivative(g_coeffs, x)

    return ((g_der_x * h_x) - (g_x * h_der_x)) / (h_x ** 2)
    