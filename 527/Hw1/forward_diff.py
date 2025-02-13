import numpy as np
import math
import matplotlib.pyplot as plt

# Decorator to ensure double precision
def double_dec(func):
    def wrapper(*args):
        args = tuple(np.double(arg) for arg in args)

        result = func(*args)

        return np.double(result)
    return wrapper

@double_dec
def f(x):
    return math.tan(x)

@double_dec
def analytical_d(x):
    return f(x) **2 + 1

@double_dec
def numerical_d(x, k):
    h = pow(10, -k)
    return (f(x+h) - f(x)) / h

@double_dec
def the_bound(x, k):
    h = pow(10, -k)
    # Second derivative is increasing
    xi = x + h
    M2 = 2 * (math.sin(xi) / pow(math.cos(xi), 3)) + 1
    return (M2 * h) / 2

err_list = []
the_bound_list = []
k_list = range(1, 17)
for k in k_list:
    err = abs(numerical_d(1., k) - analytical_d(1.))
    err_list.append(err)
    the_bound_list.append(the_bound(1., k))
    
plt.figure(figsize=(8,6))
plt.plot(k_list, err_list, linestyle='-', color = 'blue', label = 'numerical error')
plt.plot(k_list, the_bound_list, linestyle='-', color = 'red', label = 'therotical bound')
plt.xlabel('k')
plt.ylabel('Error')
plt.yscale('log')
plt.title('Error of forward difference scheme')
plt.legend()
plt.savefig('forward_diff.png')