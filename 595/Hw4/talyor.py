import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def approximation(func:sp.Expr, start: float, end: float, degree: int, fixed_c: float) -> tuple[np.ndarray, np.ndarray]:

    x = sp.symbols('x')
    # print(func)
    # f = func(x) 
    n_terms = degree + 1  # Parameter in sp.series is the numver of terms
    taylor = sp.series(func, x, fixed_c, n_terms).removeO()  # Remove higher degree terms
    taylor_func = sp.lambdify(x, taylor, 'numpy')  # Generate callable function
    n = 100
    x = np.linspace(start, end, n)
    y = taylor_func(x)

    return x, y


x = sp.symbols('x')
test_func = x * (sp.sin(x))**2 + sp.cos(x)  # Symbol function
seris_x, series_y = approximation(func=test_func, start=-10.0, end=10.0, degree=99, fixed_c=0.0)

f = sp.lambdify(x, test_func, 'numpy')  # Generate callable function
test_x = np.linspace(-10.0, 10.0, 100)
test_y = f(test_x)


plt.figure(figsize=(10,6))
plt.plot(test_x, test_y, label='Actual function', color='b')
plt.scatter(seris_x, series_y, label='Taylor approx', color='r', marker='o')
plt.xlabel('x')
plt.ylabel(f'$x sin^2(x) + cos(x)$')
plt.title('Taylor series approximation of the target function.')
plt.legend()
plt.savefig('talyor.png')