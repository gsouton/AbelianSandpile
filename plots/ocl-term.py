#!/usr/bin/env python3
from expTools import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

import os

easypapOptions = {
    "-k ": ["ssandPile"],
    "-v ": ["omp_tiled"],
    "-wt ": ["opt"],
    "-s ":  [2 ** i for i in range(5, 10)],
    "-of ": ["terminaison.csv"]
}

# OMP Internal Control Variable
ompICV = {
}

nbrun = 1

# Lancement des experiences
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")
df = pd.read_csv("terminaison.csv", sep=";")

print(df.head())

x = df['size']
y = df['iterations']

# https://www.statology.org/curve-fitting-python/

# fit polynomial models up to degree 5
model1 = np.poly1d(np.polyfit(x, y, 1))
model2 = np.poly1d(np.polyfit(x, y, 2))
model3 = np.poly1d(np.polyfit(x, y, 3))
model4 = np.poly1d(np.polyfit(x, y, 4))
model5 = np.poly1d(np.polyfit(x, y, 5))
model6 = np.poly1d(np.polyfit(x, y, 6))

models = [model1, model2, model3, model4, model5, model6]


# define function to calculate adjusted r-squared
def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1 - \
        (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))

    return results


r_values = [adjR(x, y, 1), adjR(x, y, 2), adjR(x, y, 3),
            adjR(x, y, 4), adjR(x, y, 5), adjR(x, y, 6)]

# calculated adjusted R-squared of each model
print(f"{r_values}")

max_r = r_values[0]
index = 0
for r_value in r_values:
    if r_value['r_squared'] > max_r['r_squared']:
        max_r = r_value
        index += 1

print(f"best model is model{index+1} with {max_r}")
print(models[index])


def f(x):
    return 1.49e-07 * x**4 - 0.0001171 * x**3 + 0.2898 *x**2 - 2.583 * x + 30.62


for i in range(5, 10):
    size = 2**i
    print(f"for size {size} expect around {f(size)} iterations")

# create scatterplot
polyline = np.linspace(1, 600, 50)
plt.scatter(x, y)
plt.xlabel('size')
plt.ylabel('iterations')

# add fitted polynomial lines to scatterplot
plt.plot(polyline, model1(polyline), color='green')
plt.plot(polyline, model2(polyline), color='red')
plt.plot(polyline, model3(polyline), color='purple')
plt.plot(polyline, model4(polyline), color='blue')
plt.plot(polyline, model5(polyline), color='orange')
plt.plot(polyline, model6(polyline), color='yellow')
plt.show()

