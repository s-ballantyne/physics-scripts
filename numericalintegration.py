#!/usr/bin/env python3
"""Solution to second MT2503 Python project."""

import numpy as np
import matplotlib.pyplot as plot


def trapezium_rule(f, a, b, num: int = 100):
	"""
	:param f: f(x)
	:param a: start
	:param b: end
	:param num: number of samples to pass to linspace
	:return: approximate area under f(x) in [a, b]
	"""
	x, h = np.linspace(a, b, num=num, retstep=True)

	"""
		sum(f(x[1:])) = f(x_1) + ... + f(x_n)
		sum(f(x[:-1])) = f(x_0) + ... + f(x_n-1)
		sum(f(x[1:]) + f(x[:-1])) = f(x_0) + f(x_n) + 2.(f(x_1) + ... + f(x_n-1))
		(h / 2) * (above) = h.((f(x_0) + f(x_n)) / 2 + f(x_1) + ... + f(x_n-1))
	"""
	return (h / 2.) * (sum(f(x[1:]) + f(x[:-1])))


def cringe_rule(f, a, b, num):
	h = (b - a) / n

	integral = (f(a) + f(b)) / 2.
	for i in range(1, num):
		integral = integral + f(a + i * h)

	integral = integral * h
	return integral


def exercise_one():
	f = lambda x: 1. / (3. + x**2)
	F = lambda x: np.arctan(x / np.sqrt(3)) / np.sqrt(3)

	a, b, n = -3, 3, 8
	print(f"Approximation of I using {n} intervals: {trapezium_rule(f, a, b, n)}")

	j = lambda x: np.exp(-(np.cos(x) ** 2))
	a, b, n = 0, 2, 100
	print("\n".join(f"Approximation of J using {n} intervals: {trapezium_rule(f, a, b, n)}" for n in range(10, 100, 10)))

	K = lambda x: trapezium_rule(j, 0, x, num=100)

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


exercise_one()

plot.show()
