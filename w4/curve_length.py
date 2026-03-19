#Ex5.51
import math
def calc_curve_length(points):
    length = 0
    for i in range(1, len(points)):
        dc_1 = points[i][0] - points[i-1][0]
        dc_2 = points[i][1] - points[i-1][1]

        length += math.sqrt(dc_1**2+dc_2**2)
    return length

N = 100000
points = []
for i in range(N):
    t = i/N
    x = 2*t+1
    y = -t**2
    points.append((x,y))

print(calc_curve_length(points))
#_______________________________________________________________________________________________

#Ex5.52
speed_function = lambda t: 2 * math.sqrt(1+ t**2)

points_ex52 = []
for i in range(N):
    t = i/N
    speed_function(t)
    points_ex52.append(t)
    
print(calc_curve_length(points_ex52))








#Chat

import torch

# Define the curve function c(t) from Exercise 5.3 [cite: 55]
def c(t):
    t = t.reshape(-1, 1)
    # c(t) = [2t + 1, -t^2]
    return torch.concat((2*t + 1, -t**2), dim=1)

# Define the analytical speed function ||c'(t)|| [cite: 65]
def speed(t):
    # ||c'(t)|| = 2 * sqrt(1 + t^2)
    return 2 * torch.sqrt(1 + t**2)

def compute_lengths(num_steps=1000):
    T = torch.linspace(0, 1, num_steps)
    dt = T[1] - T[0]

    # Method 1: Discrete distance between points on the curve [cite: 143]
    points = c(T)
    # Calculate difference between consecutive points [cite: 147]
    delta = points[1:] - points[:-1]
    # Sum of Euclidean norms of differences [cite: 149, 154]
    dist_length = torch.sqrt((delta**2).sum(dim=1)).sum()

    # Method 2: Integrating the speed function [cite: 156]
    s = speed(T)
    # Integral approx: sum(speed * dt) 
    int_length = s.sum() * dt

    return dist_length.item(), int_length.item()

# Execute and print results
len_discrete, len_integrated = compute_lengths()
print(f"Length (Discrete Curve): {len_discrete:.4f}")
print(f"Length (Integrated Speed): {len_integrated:.4f}")
print(f"Analytical Solution: ~2.2956") # [cite: 75]