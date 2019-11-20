# !/usr/bin/env python3
"""Solution to second MT2503 Python project."""
import numpy as np
import matplotlib.pyplot as plot


def trapezium_rule_1d(f, a, b, n):
	"""
	:param f: function to integrate
	:param a: start
	:param b: end
	:param n: number of trapeziums
	:return: approximate area under f(x) in [a, b]
	"""
	h = (b - a) / n

	area = (f(a) + f(b)) / 2
	area += sum(f(a + i * h) for i in range(1, n))

	return area * h

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
	return trapezium_rule_1d(lambda y: trapezium_rule_1d(lambda x: f(x, y), a, b, nx), c, d, ny)

def exercise_one():
	""""""
	"""Part (a)"""
	f = lambda x: 1. / (3. + x ** 2)
	F = lambda x: np.arctan(x / np.sqrt(3)) / np.sqrt(3)

	a, b, n = -3, 3, 8
	actual = F(b) - F(a)
	approx = trapezium_rule_1d(f, a, b, n)
	error = np.abs(approx - actual)
	print(f"Exact value of I   : {actual}")
	print(f"Approx. value of I : {approx}")
	print(f"Absolute error in I: {error}")

	"""Part (b)"""
	for h in np.logspace(0, -4, num=5, base=10):
		n = int((b - a) / h)
		error = np.abs(trapezium_rule_1d(f, a, b, n) - actual)
		print(f"h = {h:.4f}, n = {n}, error = {error:f}")

	print("Based on the above results, the error is proportional to the step size cubed (p = 3).\n"
		  "Changing the width by a factor of ten causes the error to shift by three decimal places (10^3 == 1000).")

	"""Part (c)"""
	j = lambda x: np.exp(-(np.cos(x) ** 2))
	a, b = 0, 2

	for n in range(10, 200, 20):
		print(f"n = {n} J = {trapezium_rule_1d(j, a, b, n):.4f}")

	print("After the number of trapeziums reaches 70, the value of J remains as 1.4183.")

	"""Part (d)"""
	K = lambda x: trapezium_rule_1d(j, 0, x, 100)

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
	f = lambda x, y: np.exp(-(x ** 2 + y ** 2))

	# [-inf, inf] was taken to be [-10, 10]
	# since exp(-(10^2)) is small anyway.
	nx, ny = 20, 20
	actual = np.pi
	approx = trapezium_rule_2d(
		f,
		a=-10, b=10,
		c=-10, d=10,
		nx=nx, ny=ny
	)

	print(
		"∫∫exp(-(x^2 + y^2)).dx.dy between x = -inf -> inf, y = -inf, inf:\n"
		f" Approx. value : {approx:f} (nx = {nx}, ny = {ny})\n"
		f" Actual value  : {actual:f} (π)\n"
		f" Absolute error: {np.abs(approx - actual):f}\n"
		f" Conclusions: trapezium_rule_2d is fairly accurate for smooth functions."
	)


if __name__ == "__main__":
	exercise_one()
	exercise_two()

	plot.show()
