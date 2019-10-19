#!/usr/bin/env python3
"""Solution to first MT2503 Python project."""

import numpy as np
import matplotlib.pyplot as plot


try:
	has_scipy = True
	from scipy.misc import derivative as scipy_derivative
except ImportError:
	has_scipy = False
	scipy_derivative = lambda: 0
	pass


def forward_difference(f, x, h):
	"""Forward difference method"""
	return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
	"""Central difference method"""
	return (f(x + h) - f(x - h)) / (2 * h)


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

	"""retstep=True makes linspace also return the step size."""
	x, h = np.linspace(a, b, num=n, endpoint=endpoint, retstep=True)

	y = f(x)
	dy_dx = diff_func(f, x, h)

	return dy_dx, y, x


def exercise_one():
	"""
	Exercise one
	"""

	"""Create a new figure for plotting results"""
	figure = plot.figure()
	axes = figure.add_subplot(111)

	"""
	Define f(x) and f'(x)
	"""
	f = lambda x: np.exp(-x * x)
	g = lambda x: -2 * x * f(x)

	"""Evaluate forward difference"""
	dy_dx, y, x = finite_difference(f, -2, 2, forward_difference)

	"""Evaluate and find maximum error"""
	err = g(x) - dy_dx
	abs_err = np.abs(err)
	x_max, abs_err_max = np.amax(abs_err), x[np.argmax(abs_err)]

	"""Print maximum error, plot error on graph"""
	print(f"x_max = {x_max:.12f} err = {abs_err_max:.15f}")
	axes.plot(x, abs_err, label="error", color="r")

	"""
	https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html
	Create logspace (similar linspace but log-scaled) for the step-size h between a and b
	"""
	a, b = -1, -12
	h = np.logspace(a, b, num=np.abs(b - a + 1), base=10)

	""""""
	err = g(x_max) - forward_difference(f, x_max, h)
	abs_err = np.abs(err)

	"""Print results to screen"""
	print("\n".join(f"step  = {h:.12f} err = {e:.15f}" for h, e in zip(h, abs_err)))
	print(	f"The error in the result decreases to a minimum of "
			f"{np.amin(abs_err):.15f} (h = {h[np.argmin(abs_err)]}).\n"
			f"After that, it appears to increase as the step size gets smaller.\n"
			f"Perhaps this is due to the accumulation of floating-point arithmetic errors,"
			f" since real numbers cannot be exactly represented on a computer with finite"
			f" precision, which may be magnified by the representation of the mathematical"
			f" formula in this script.")

	# axes.plot(x, y, label="$y = f(x)$", color="b")
	# axes.errorbar(x, dy_dx, err, label="$y = f^{\prime}(x)$", color="g")
	# axes.scatter(x_max, abs_err_max, label="maximum error", color="black", marker="o")

	axes.set_title(r"Exercise one: $f(x) = e^{-x^{2}}$")
	axes.grid()
	axes.legend()


def exercise_two():
	"""
	Exercise two
	"""

	"""Define f(x)"""
	f = lambda x: np.tan(x * np.sin(x))

	"""Evaluate the derivative using second-order central difference"""
	dy_dx, y, x = finite_difference(f, -1, 1, central_difference)

	"""Print results for (b)"""
	x_i = -0.5
	h = 1e-4
	if has_scipy:
		"""SciPy uses the same method."""
		print(f"dy/dx at x = {x_i}, step = {h}, using SciPy's derivative: {scipy_derivative(f, x_i, h):.4f}")

	central = central_difference(f, x_i, h)
	print(f"dy/dx at x = {x_i}, step = {h}, using central difference: {central:.4f}")

	dh = 1
	forward = forward_difference(f, x_i, h / dh)
	while round(forward, 5) != round(central, 5) and dh < 20:
		dh += 1
		forward = forward_difference(f, x_i, h / (10 ** dh))

	print(f"dy/dx at x = {x_i}, step = {h / (10 ** dh)}, using forward difference: {forward:.4f}")
	print(f"Forward difference has to have a step size {dh} order(s) of magnitude less than required for the central"
		  f" difference method to get identical results.")

	"""Create a new figure and plot the results"""
	figure = plot.figure()
	axes = figure.add_subplot(111)

	axes.plot(x, y, label=r"$y = f(x)$", color="b")
	axes.plot(x, dy_dx, label=r"$y = f^{\prime}(x)$", color="g")

	axes.set_title(r"Exercise two: $f(x) = \tan(x \cdot \sin(x))$")
	axes.grid()
	axes.legend()


def grad(f, coords, steps):
	"""
	Calculates the gradient of function f for each coordinate in :param coords: with step sizes in :param steps:

	:param f: function to calculate derivative of
	:param coords: tuple of coordinates of interest (x, y, etc.)
	:param steps: step sizes for each coordinate. Must be same size as :param coords:
	:return: 2nd order derivative approximation for each coordinate

	Pseudo-code for little babies:
		for i in range(n):
			h = steps[i]
			df_by_dx_i = (f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_i - h, ..., x_n)) / (2 * h)
			out.append(df_by_dx_i)
	"""
	return [(f(*(coords[:i]+(coord+step,)+coords[i+1:])) - f(*(coords[:i]+(coord-step,)+coords[i+1:]))) / (2 * step) for i, (coord, step) in enumerate(zip(coords, steps))]


def directional_derivative(f, coords, vector, steps):
	"""
	Directional derivative of f is:
		grad(f) dot (direction vector as a unit vector)

	:param f:
	:param coords:
	:param vector:
	:param steps:
	:return:
	"""
	return np.dot(grad(f, coords, steps), vector / np.linalg.norm(vector))


def exercise_three():
	"""
	Exercise three
	"""

	"""Define f(x, y)"""
	f = lambda x, y: np.sin(x * np.arccos(y))

	print("My scheme is: \n\t(f(x + h, y) - f(x - h, y)) / 2h\n\t(f(x, y + k) - f(x, y - k)) / 2k")
	print(f"Directional derivative: {directional_derivative(f, coords=(1., 0.5), vector=(2., -1.), steps=(0.001, 0.001))}")


if __name__ == "__main__":
	exercise_one()
	exercise_two()
	exercise_three()

	plot.show()
