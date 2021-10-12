import math
import numpy as np


def diff_angle(target, source):
    a = target - source
    a = (a + 3.14159) % (2*3.14159) - 3.14159
    return a


def any_diff_weights(start, end, ndiv):
    # idea similar to central_diff_weights, but not necessarily central
    # both bounds (start and end) are included
    # n_points = 1+(end-start)
    n_points = 1+(end-start)
    assert ndiv < n_points

    A = np.array([np.arange(start, 1+end) ** k / math.factorial(k) for k in range(n_points)]).T
    #A*(f, f', f'',...) = (f(x-h), f(x), f(x+h)...)
    return np.linalg.inv(A)[ndiv,:]

def any_diff_weights_no_zero(start, end, ndiv):  
    # both bounds (start and end) are included
    # n_points = 1+(end-start)
    n_points = (end-start)
    assert ndiv < n_points

    A = np.array([np.hstack((np.arange(start, 0), np.arange(1, 1+end))) ** k / math.factorial(k) for k in range(n_points)]).T
    #A*(f, f', f'',...) = (f(x-h), f(x), f(x+h)...)
    return np.linalg.inv(A)[ndiv,:]



