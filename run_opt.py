import numpy as np 
from ga_xgb import GA

def rosenbrock(x, a=1, b=100):
    """
    N-dimensional Rosenbrock function.

    Parameters:
    x (numpy.ndarray): An n-dimensional input array.
    a (float): The first coefficient of the Rosenbrock function (default is 1).
    b (float): The second coefficient of the Rosenbrock function (default is 100).

    Returns:
    float: The value of the Rosenbrock function at the given input.
    """
    n = len(x)
    result = 0
    for i in range(n - 1):
        result += a * (x[i] - x[i - 1] ** 2) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2
    return result

pop_size = 100
generations = 25
ga = GA(pop_size, generations, rosenbrock, lb=-2, ub=2, n_vars=15, oversize_mult=3, n_best=15)
score, ind = ga.run()
print(score, ind)