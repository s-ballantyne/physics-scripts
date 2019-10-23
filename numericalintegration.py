#!/usr/bin/env python3
"""Solution to second MT2503 Python project."""

import numpy as np
import matplotlib.pyplot as plot


def ndm(*args, **kwargs):
	"""
	Adapted from https://stackoverflow.com/a/22778484

	Similar effect to np.meshgrid(...) (https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html)

	:param args:
	:return:
	"""
	start = kwargs.get("start", 0)
	stop = kwargs.get("stop", None)
	return [x[(np.newaxis,) * i + (slice(start, stop),) + (np.newaxis,) * (len(args) - i - 1)] for i, x in enumerate(args)]


"""
args = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
print(*[x[(np.newaxis,) * i + (slice(np.newaxis),) + (np.newaxis,) * (len(args) - i - 1)] for i, x in enumerate(args)])
print(*[x[(np.newaxis,) * i + (slice(1, -1),) + (np.newaxis,) * (len(args) - i - 1)] for i, x in enumerate(args)])
print(*[x[(np.newaxis,) * i + (slice(-1, None),) + (np.newaxis,) * (len(args) - i - 1)] for i, x in enumerate(args)])
"""

def trapezium_rule_a(y, h):
	""""""
	"""
		sum(f(x[1:])) = f(x_1) + ... + f(x_n)
		sum(f(x[:-1])) = f(x_0) + ... + f(x_n-1)
		sum(f(x[1:]) + f(x[:-1])) = f(x_0) + f(x_n) + 2.(f(x_1) + ... + f(x_n-1))
		(h / 2).(above) = h.((f(x_0) + f(x_n)) / 2 + f(x_1) + ... + f(x_n-1))
	"""

	return ((h / 2.) * (np.sum(y[1:, np.newaxis] + y[:-1, np.newaxis], axis=0)))[0]


def trapezium_rule_1d(f, a, b, num=100):
	"""
	:param f: f(x)
	:param a: start
	:param b: end
	:param num: number of samples to pass to linspace
	:return: approximate area under f(x) in [a, b]
	"""
	x, h = np.linspace(a, b, num=num, retstep=True)
	return trapezium_rule_a(f(x), h)


def trapezium_rule_2d(f, a, b, c, d, nx=100, ny=100):
	x, h = np.linspace(a, b, num=nx, retstep=True)
	y, k = np.linspace(c, d, num=ny, retstep=True)
	return trapezium_rule_a(trapezium_rule_a(f(*ndm(x, y)), h), k)


print(f"area = {trapezium_rule_2d(lambda x, y: 1. / (x**4 + y**2), np.sqrt(2), 3, 0, 8)}")


def exercise_one():
	f = lambda x: 1. / (3. + x**2)
	F = lambda x: np.arctan(x / np.sqrt(3)) / np.sqrt(3)

	a, b, n = -3, 3, 8
	print(f"Approximation of I using {n} intervals: {trapezium_rule_1d(f, a, b, n)}")

	j = lambda x: np.exp(-(np.cos(x) ** 2))
	a, b, n = 0, 2, 100
	print("\n".join(f"Approximation of J using {n} intervals: {trapezium_rule_1d(j, a, b, n)}" for n in range(10, 100, 10)))

	K = lambda x: trapezium_rule_1d(j, 0, x, num=100)

	figure = plot.figure()
	axes = figure.add_subplot(111)

	x = np.linspace(0, 5 * np.pi)
	axes.plot(x, j(x), label="y = j(x)")
	axes.plot(x, K(x), label="y = K(x)")

	axes.set_title("Exercise 1. (d)")
	axes.set_xlabel("x")
	axes.set_ylabel("y")
	axes.grid()
	axes.legend()


def exercise_two():
	pass


exercise_one()

plot.show()
