import matplotlib.pyplot as plt
import numpy as np

# Generate some data for the subplots
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2)

# Flatten the array of subplots into a 1D array
"""Note that the axes in this case is an array of 4 objects ,each object with a function plot(x,y) that accepts x and 
y 1D arrays  and tries to plot them against each other in teh main FIGURE the flatten function takes [[sub_plot_1, 
sub_plot_2],[sub_plot_3,sub_plot_4]] and returns a stack containing [sub_plot_1, sub_plot_2,sub_plot_3,sub_plot_4]
we then enumerate the same , so that we can use the  index to access and show the title while keeping the Code DRY"""

axes_flat = axes.flatten()
titles = ['Subplot 1', 'Subplot 2', 'Subplot 3', 'Subplot 4']

for index, item in enumerate(axes_flat):
    item.plot(x, y)
    item.set_title(titles[index])

if __name__ == '__main__':
    plt.show()
