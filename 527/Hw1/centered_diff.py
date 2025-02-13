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

# Decorator to ensure single precision
def single_dec(func):
    def wrapper(*args):
        args = tuple(np.half(arg) for arg in args)

        result = func(*args)

        return np.half(result)
    return wrapper

@double_dec
def db_f(x):
    return math.tan(x)

@double_dec
def db_analytical_d(x):
    return db_f(x) **2 + 1

@double_dec
def db_numerical_d(x, k):
    h = pow(10, -k)
    return (db_f(x+h) - db_f(x-h)) / (2*h)

@single_dec
def sg_f(x):
    return math.tan(x)

@single_dec
def sg_analytical_d(x):
    return sg_f(x) **2 + 1

@single_dec
def sg_numerical_d(x, k):
    h = pow(10, -k)
    return (sg_f(x+h) - sg_f(x-h)) / (2*h)

@double_dec
def the_bound(x, k):
    h = pow(10, -k)
    # Second derivative is increasing
    xi = x + h
    M2 = 2 * ( (1+ 2*pow(math.sin(xi),2)) / pow(math.cos(xi), 4) ) + 1
    return (M2 * pow(h, 2)) / 6

db_err_list = []
the_bound_list = []
k_list = range(1, 17)
for k in k_list:
    err = abs(db_numerical_d(1., k) - db_analytical_d(1.))
    db_err_list.append(err)
    the_bound_list.append(the_bound(1., k))

sg_err_list = []
k_list = range(1, 17)
for k in k_list:
    err = abs(sg_numerical_d(1., k) - sg_analytical_d(1.))
    sg_err_list.append(err)

plt.figure(figsize=(8,6))
plt.plot(k_list, db_err_list, linestyle='-', color='blue', label='double')
plt.plot(k_list, sg_err_list, linestyle='-', color='purple', label='single')
plt.plot(k_list, the_bound_list, linestyle='-', color='red', label='therotical')
plt.xlabel('k')
plt.ylabel('Error')
plt.yscale('log')
plt.title('Error of centered difference scheme')
plt.legend()
plt.savefig('centered_diff.png')