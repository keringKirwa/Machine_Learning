from math import pi, sin
from random import random, uniform

points_inside_the_curve = 0
N = 10000
for i in range(N):
    point = (uniform(0, pi), uniform(0, 1))
    if sin(point[0]) >= point[1]:
        points_inside_the_curve += 1

answer = (float(points_inside_the_curve) / float(N)) * pi

# Call function
if __name__ == '__main__':
    print("The sine of random integers between 0 and pi is {}", format(answer))
