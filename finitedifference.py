#!/usr/bin/env python3
"""Solution to first MT2503 Python project."""

import numpy as np
import matplotlib.pyplot as plot

try:
	has_scipy = True
	from scipy.misc import derivative as scipy_ndiff
except ImportError:
	has_scipy = False
	scipy_ndiff = lambda: 0
	pass


def amax2(array_like, index_lookup):
	"""Convenience function to find maximum in a numpy array and lookup from a second array at the same time"""
	i = np.argmax(array_like)
	return index_lookup[i], array_like[i]


def forward_difference(f, x, h):
	"""Forward difference method"""
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

	"""retstep=True makes linspace also return the step size."""
	x, h = np.linspace(a, b, num=n, endpoint=endpoint, retstep=True)

	y = f(x)
	dy_dx = diff_func(f, x, h)

	return dy_dx, y, x


def exercise_one():
	"""
	Exercise one
	"""

	"""
	Define f(x) and f'(x)
	"""
	f = lambda x: np.exp(-x * x)
	fp = lambda x: -2 * x * np.exp(-x * x)

	"""Evaluate forward difference"""
	dy_dx, y, x = finite_difference(f, -2, 2, forward_difference)

	err = fp(x) - dy_dx
	abs_err = np.abs(err)

	x_max, abs_err_max = amax2(abs_err, x)
	print(f"x_max = {x_max:.12f} err = {abs_err_max:.15f}")

	h_space = np.logspace(-1, -12, num=12, base=10)
	new_err = fp(x_max) - forward_difference(f, x_max, h_space)
	abs_new_err = np.abs(new_err)

	print("\n".join(f"step  = {h:.12f} err = {e:.15f}" for e, h in zip(abs_new_err, h_space)))
	print(	f"The error in the result decreases to a minimum of "
			f"{np.amin(abs_new_err):.15f} (h = {h_space[np.argmin(abs_new_err)]}).\n"
			f"After that, it appears to increase as the step size gets smaller.\n"
			f"Perhaps this is due to the accumulation of floating-point arithmetic errors,"
			f" since real numbers cannot be exactly represented on a computer with finite"
			f" precision, which may be magnified by the representation of the mathematical"
			f" formula in this script.")

	figure = plot.figure()
	axes = figure.add_subplot(111)

	axes.plot(x, y, label="$y = f(x)$", color="b")
	axes.errorbar(x, dy_dx, err, label="$y = f^{\prime}(x)$", color="g")
	axes.plot(x, err, label="error", color="r")
	axes.scatter(x_max, abs_err_max, label="maximum error", color="black", marker="o")

	axes.set_title(r"Exercise one: $f(x) = e^{-x^{2}}$")
	axes.grid()
	axes.legend()


def central_difference(f, x, h):
	"""Central difference method"""
	return (f(x + h) - f(x - h)) / (2 * h)


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
		"""Uses the central difference method."""
		print(f"dy/dx at x = {x_i}, step = {h}, using SciPy's derivative: {scipy_ndiff(f, x_i, h):.4f}")

	central = central_difference(f, x_i, h)
	print(f"dy/dx at x = {x_i}, step = {h}, using central difference: {central:.4f}")

	dh = 1
	forward = forward_difference(f, x_i, h / dh)
	while round(forward, 5) != round(central, 5) and dh < 20:
		dh += 1
		forward = forward_difference(f, x_i, h / (10 ** dh))

	print(f"dy/dx at x = {x_i}, step = {h}, using forward difference: {forward:.4f}")
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


def exercise_three():
	"""
	Exercise three:
		f(x) = cos(pi * exp(-x))

	"""

	"""Define f(x, y)"""
	f = lambda x, y: np.sin(x * np.arccos(y))

	scheme_x = "f_x(x, y) = (f(x + h, y) - f(x - h, y)) / 2h"
	scheme_y = "f_y(x, y) = (f(x, y + k) - f(x, y - k)) / 2k"
	print(f"My scheme is:\n\t{scheme_x}\n\t{scheme_y}")




if __name__ == "__main__":
	exercise_one()
	exercise_two()
	exercise_three()

	plot.show()
