#!/usr/bin/env python3
"""Solution to first MT2503 Python project."""

import numpy as np
import matplotlib.pyplot as plot


def amax2(array_like, index_lookup):
	i = np.argmax(array_like)
	return index_lookup[i], array_like[i]


def forward_difference(f, x, h):
	return (f(x + h) - f(x)) / h


def finite_difference(f, a, b, diff_func: callable, n: int = 100, endpoint: bool = True):
	"""
	:param f: function f(x) to evaluate
	:param a: start point of interval
	:param b: end point of interval
	:param diff_func: forward_difference/central_difference etc
	:param n: number of samples to generate.
	:param endpoint: if True, :param b: is the last point. Otherwise, it is not included.
	:return: f'(x), f(x) over [a, b], [a, b]
	"""
	x, h = np.linspace(a, b, num=n, endpoint=endpoint, retstep=True)

	y = f(x)
	dy_dx = diff_func(f, x, h)

	return dy_dx, y, x


def exercise_one():
	f = lambda _x: np.exp(-_x * _x)
	fp = lambda _x: -2 * _x * np.exp(-_x * _x)

	dy_dx, y, x = finite_difference(f, -2, 2, forward_difference)

	err = fp(x) - dy_dx
	abs_err = np.abs(err)

	x_max, abs_err_max = amax2(abs_err, x)
	print(f"x_max = {x_max:.12f} err = {abs_err_max}")

	for h in np.logspace(-1, -12, num=12, base=10):
		print(f"step  =  {h:.12f} err = {fp(x_max) - forward_difference(f, x_max, h):.17f}")

	print("""todo: insert comment here""")

	figure = plot.figure()
	axes = figure.add_subplot(111)

	axes.plot(x, y, label="y = f(x)", color="b")
	axes.errorbar(x, dy_dx, err, label="y = f'(x)", color="g")
	axes.plot(x, err, label="error", color="r")
	axes.scatter(x_max, abs_err_max, label="maximum error", color="black", marker="o")

	axes.set_title("Exercise one: f(x) = exp(-x*x)")
	axes.grid(True)
	axes.legend()


def central_difference(f, x, h):
	return (f(x + h) - f(x - h)) / (2 * h)


def exercise_two():
	f = lambda _x: np.tan(_x * np.sin(_x))

	dy_dx, y, x = finite_difference(f, -1, 1, central_difference)

	print(f"dy/dx at x = 0.5, using central difference: {central_difference(f, 0.5, 1e-6)}")
	print(f"dy/dx at x = 0.5, using forward difference: {forward_difference(f, 0.5, 1e-6)}")
	print("""todo: insert comment here""")

	figure = plot.figure()
	axes = figure.add_subplot(111)

	axes.plot(x, y, label="y = f(x)", color="b")
	axes.plot(x, dy_dx, label="y = f'(x)", color="g")
	axes.set_title("Exercise two: f(x) = tan(x sin(x))")
	axes.grid(True)
	axes.legend()


def exercise_three():
	pass


if __name__ == "__main__":
	# exercise_one()
	# exercise_two()
	exercise_three()

	plot.show()
