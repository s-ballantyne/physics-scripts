#!/usr/bin/env python3
"""Solution to first MT2503 Python project."""

import numpy as np
import matplotlib.pyplot as plot


def forward_difference(f, x, h):
	"""Forward difference method"""
	return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
	"""Central difference method"""
	return (f(x + h) - f(x - h)) / (2 * h)


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
	x, h = np.linspace(-2, 2, num=100, retstep=True)
	dy_dx = forward_difference(f, x, h)

	"""Evaluate and find maximum error"""
	err = dy_dx - g(x)
	abs_err = np.abs(err)
	x_max, abs_err_max = x[np.argmax(abs_err)], np.amax(abs_err)

	"""Print maximum error, plot error on graph"""
	print(f"x_max = {x_max:.12f} error = {abs_err_max}")
	axes.plot(x, abs_err, label="error", color="r")

	"""
	https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html
	Create logspace (similar linspace but log-scaled) for the step-size h between a and b
	"""
	a, b = -1, -12
	num = np.abs(b - a - 1)
	h = np.logspace(a, b, num=num)

	""""""
	err = g(x_max) - forward_difference(f, x_max, h)
	abs_err = np.abs(err)

	"""Print results to screen"""
	print("\n".join(f"step = {h:.{num}f} error = {e:.{num}f} error/step = {e / h:.{num}g}" for h, e in zip(h, abs_err)))
	print(	f"The error in the result decreases to a minimum of "
			f"{np.amin(abs_err):.{num}f} (h = {h[np.argmin(abs_err)]:.{num}f}).\n"
			f"After that, it appears to increase as the step size gets smaller.\n"
			f"Perhaps this is due to the accumulation of floating-point arithmetic errors,\n"
			f" since real numbers cannot be exactly represented on a computer with finite\n"
			f" precision, which may be magnified by the representation of the mathematical\n"
			f" formula in this script.\n"
			f"Additionally, it appears the the error is directly proportional to the step size h,\n"
			f" until it reaches the minimum value (error / step is roughly constant).")

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

	"""Create a new figure for plotting the results"""
	figure = plot.figure()
	axes = figure.add_subplot(111)

	"""Define f(x)"""
	f = lambda x: np.tan(x * np.sin(x))

	"""Evaluate the derivative using second-order central difference"""
	x, h = np.linspace(-1, 1, num=100, retstep=True)
	dy_dx = central_difference(f, x, h)

	"""Plot results for (a)"""
	axes.plot(x, f(x), label=r"$y = f(x)$", color="b")
	axes.plot(x, dy_dx, label=r"$y = f^{\prime}(x)$", color="g")

	"""Print results for (b)"""
	x_i = -.5
	h = 1e-4

	central = central_difference(f, x_i, h)
	print(f"dy/dx at x = {x_i}, step = {h:f}, using central difference: {central:.4f}")

	dh = 1
	forward = forward_difference(f, x_i, h / dh)
	while round(forward, 5) != round(central, 5) and dh < 20:
		dh += 1
		forward = forward_difference(f, x_i, h / (10 ** dh))

	print(f"dy/dx at x = {x_i}, step = {h / (10 ** dh):f}, using forward difference: {forward:.4f}")
	print(f"Forward difference has to have a step size {dh} order(s) of magnitude\n"
		  f" less than required for the central difference method to get results\n"
		  f" identical to 4 decimal places.")

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

	f_x = (f(x + h, y) - f(x - h, y)) / 2h
	f_y = (f(x, y + k) - f(x, y - k)) / 2k

	Equivalent to:
		for i in range(1, n):
			df/dx_i = (f(x_1, ..., x_i + h_i, ..., x_n) - f(x_1, ..., x_i - h_i, ..., x_n)) / (2 * h_i)
			out.append(df/dx_i)

	Note: for single variable f, it should be called like: grad(f, (x,), (h,))
	"""
	return [(f(*(coords[:i]+(coords[i]+step,)+coords[i+1:])) - f(*(coords[:i]+(coords[i]-step,)+coords[i+1:]))) / (2 * step) for i, step in enumerate(steps)]


def directional_derivative(f, coords, direction, steps):
	"""
	Directional derivative of f is:
		grad(f) dot (direction vector as a unit vector)

	:param f:
	:param coords:
	:param direction:
	:param steps:
	:return:
	"""
	return np.dot(grad(f, coords, steps), direction / np.linalg.norm(direction))


def exercise_three():
	"""
	Exercise three
	"""

	"""Define f(x, y)"""
	f = lambda x, y: np.sin(x * np.arccos(y))
	df_dx = lambda x, y: np.arccos(y) * np.cos(x * np.arccos(y))
	df_dy = lambda x, y: -x * np.cos(x * np.arccos(y)) / np.sqrt(1 - y * y)

	print("My scheme is:\n f_x = (f(x + h, y) - f(x - h, y)) / 2h\n f_y = (f(x, y + k) - f(x, y - k)) / 2k")

	point = (1., .5)
	direction = (2., -1.)
	steps = (1e-3, 1e-3)
	gradient = grad(f, coords=point, steps=steps)
	print(f"grad(f) at {point}: ({gradient[0]:f}, {gradient[1]:f})")
	print(f"grad(f) at {point}, along {direction}:"
		  f" {directional_derivative(f, coords=point, direction=direction, steps=steps):f}")

	"""Value determined by the exact derivative"""
	actual = np.array((df_dx(*point), df_dy(*point)))

	"""Get error for a range of step sizes, h = 1e-1, ..., 1e-5"""
	h_space = np.logspace(-1, -5, base=10, num=5)
	approx = (lambda h: grad(f, coords=point, steps=(h, h)))(h_space)

	for h, (approx_x, approx_y) in zip(h_space, np.nditer(approx)):
		error_x, error_y = np.abs(actual[0] - approx_x), np.abs(actual[1] - approx_y)
		print(f"h = {h:.{h_space.size}f} error = ({error_x:.12f}, {error_y:.12f}) error/step^2 = ({error_x/h**2:f}, {error_y/h**2:f})")

	print("It is clear from the above results that the error is proportional to the step size squared.")

if __name__ == "__main__":
	exercise_one()
	exercise_two()
	exercise_three()

	plot.show()
