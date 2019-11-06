#!/usr/bin/env python3
"""Solution to second MT2503 Python project."""
import functools

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


def _trapezium_rule(y, h):
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
	return _trapezium_rule(f(x), h)


def trapezium_rule_2d(f, a, b, c, d, nx=100, ny=100):
	"""

	:param f: function to integrate
	:param a: lower limit for x
	:param b: upper limit for x
	:param c: lower limit for y
	:param d: upper limit for y
	:param nx: number of samples for x
	:param ny: number of samples for y
	:return: area estimated by double trapezium rule
	"""
	x, h = np.linspace(a, b, num=nx, retstep=True)
	y, k = np.linspace(c, d, num=ny, retstep=True)

	"""Evaluate the trapezium rule twice"""
	return _trapezium_rule(
		_trapezium_rule(
			f(*ndm(x, y)),
			h),
		k
	)


def trapezium_rule_nd(f, *limits, num=100):
	return functools.reduce(
		_trapezium_rule,
		[(b - a) / num for a, b in limits],
		f(
			*ndm(
				*[np.linspace(a, b, num=num) for a, b in limits]
			)
		)
	)


def exercise_one():
	""""""
	"""Part (a)"""
	f = lambda x: 1. / (3. + x**2)
	F = lambda x: np.arctan(x / np.sqrt(3)) / np.sqrt(3)

	a, b, n = -3, 3, 8
	actual = F(b) - F(a)
	approx = trapezium_rule_1d(f, a, b, n)
	error = np.abs(approx - actual)
	print(f"Exact value of I   : {actual}")
	print(f"Approx. value of I : {approx}")
	print(f"Absolute error in I: {error}")

	"""Part (b)"""
	h = (b - a) / n
	print(f"p = log(error) / log(step size) = {np.log(error) / np.log(h)}")
	print(f"step size = {h}")

	"""Part (c)"""
	j = lambda x: np.exp(-(np.cos(x) ** 2))
	a, b = 0, 2
	print("\n".join(f"Approx. value of J ({n} intervals): {trapezium_rule_1d(j, a, b, num=n)}" for n in range(200, 600, 100)))

	n = 300
	print(f"J = {trapezium_rule_1d(j, a, b, n):.4f} ({n} intervals)")

	"""Part (d)"""
	K = lambda x: trapezium_rule_1d(j, 0, x, num=500)

	"""Create a new figure and axes"""
	figure = plot.figure()
	axes = figure.add_subplot(111)

	"""Plot K over [0, 5π]"""
	x = np.linspace(0, 5 * np.pi)
	# axes.plot(x, j(x), label="y = j(x)")
	axes.plot(x, K(x), label="y = K(x)")

	axes.set_title("Exercise 1. (d)")
	axes.set_xlabel("x")
	axes.set_ylabel("y")
	axes.grid()
	axes.legend()


def exercise_two():
	f = lambda x, y: np.exp(-(x**2 + y**2))

	# [-inf, inf] was taken to be [-10, 10]
	# since exp(-(10^2)) is small anyway.
	nx, ny = 50, 50
	actual = np.pi
	approx = trapezium_rule_2d(
		f,
		a=-10, b=10,
		c=-10, d=10,
		nx=nx, ny=ny
	)

	print(
		"∫∫exp(-(x^2 + y^2)).dx.dy between x = -inf -> inf, y = -inf, inf:\n"
		f" Approx value: {approx:.12f} ({nx}, {ny} samples for each integral)\n"
		f" Actual value: {actual:.12f} (π)"
	)


if __name__ == "__main__":
	exercise_one()
	exercise_two()

	plot.show()
