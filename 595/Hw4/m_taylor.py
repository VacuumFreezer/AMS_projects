import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time
import pandas as pd

# The original 'degree' input is unnecessary here
def approximation(func:sp.Expr, start: float, end: float, initial_degree: int, final_degree: int, degree_step: int, fixed_c: float) \
    -> tuple[np.ndarray, np.ndarray]:
    n = 100  #  Number of points 
    x = sp.symbols('x')
    f = sp.lambdify(x, func, 'numpy')  # Generate callable function
    x_value = np.linspace(start, end, n)
    truth_y = f(x_value)
    diff_list = []
    time_list = []

    for degree in range(initial_degree, final_degree+1, degree_step):
        x = sp.symbols('x')
        n_terms = degree + 1

        start_time = time.time()
        taylor = sp.series(func, x, fixed_c, n_terms).removeO()
        taylor_func = sp.lambdify(x, taylor, 'numpy')  # Generate callable function
        y = taylor_func(x_value)
        diff = np.linalg.norm(y - truth_y, ord=1)  # calculate the difference using 1-norm
        diff_list.append(diff)
        end_time = time.time()

        t = end_time - start_time
        time_list.append(t)

    return diff_list, time_list

x = sp.symbols('x')
target_func = x * (sp.sin(x))**2 + sp.cos(x)
difference, comsumed_time = approximation(func=target_func, start=-10.0, end=10.0, initial_degree=50, final_degree=100,
                                          degree_step=10, fixed_c=0.0)
# Save into pandas dataframe
taylor_df = pd.DataFrame({"Order": range(50, 101, 10), "Difference": difference, "Time (s)": comsumed_time})
taylor_df.to_csv('taylor_values.csv', index=False)