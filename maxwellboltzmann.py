#!/usr/bin/env python3
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plot

parser = argparse.ArgumentParser()

plot_properties = parser.add_argument_group("Plot properties")
plot_properties.add_argument("--cdf", action="store_true", help="Plot cumulative distribution instead")

physical_constants = parser.add_argument_group("Physical constants")
physical_constants.add_argument("-k", "--boltzmann", default=1.38064852e-23, help="Boltzmann's constant")

particle_properties = parser.add_argument_group("Particle properties")
particle_properties.add_argument("-m", "--mass", default=[2.325871e-26], nargs="*", help="Particle mass (kg)")
particle_properties.add_argument("-t", "--temperature", default=[250, 500, 750, 1000, 1250], nargs="*", help="Temperature (K)")

particle_properties.add_argument("-u", "--speed-min", default=0, help="Lower bound of particle speed (m/s)")
particle_properties.add_argument("-v", "--speed-max", default=4000, help="Upper bound of particle speed (m/s)")

args = parser.parse_args()

assert args.mass and args.temperature
assert all(m > 0 for m in args.mass)
assert all(t > 0 for t in args.temperature)
assert args.boltzmann > 0
assert args.speed_min >= 0
assert args.speed_min < args.speed_max

kB = args.boltzmann
speeds = np.linspace(args.speed_min, args.speed_max, args.speed_max - args.speed_min)


def pdf(v, m, T, k = kB):
	a = np.sqrt(2 / np.pi)
	scale = m / (k * T)
	v2 = v ** 2
	return a * (scale ** 1.5) * v2 * np.exp(-v2 * scale / 2)


masses_same = len(set(args.mass)) == 1
temps_same = len(set(args.temperature)) == 1

if not masses_same and not temps_same:
	format_label = lambda m, T: f"Mass: {m} kg, temperature: {T} K"
elif masses_same:
	format_label = lambda m, T: f"Temperature: {T} K"
elif temps_same:
	format_label = lambda m, T: f"Mass: {m} kg"
else:
	format_label = lambda m, T: "???"


fill = args.mass[-1] if len(args.mass) < len(args.temperature) else args.temperature[-1]
for mass, temperature in itertools.zip_longest(args.mass, args.temperature, fillvalue=fill):
	plot.plot(speeds, pdf(speeds, mass, temperature), label = format_label(mass, temperature))

plot.title(f"Maxwell-Boltzmann speed distribution for speeds {args.speed_min} to {args.speed_max} m/s")
plot.xlabel("Speed (m/s)")
plot.ylabel("Probability density (s/m)")
plot.grid(True)
plot.legend()
plot.show()
