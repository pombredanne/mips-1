import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

with open('output.txt', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    inlist = list(reader)

# ds = data series
time_ds = 8
memory_ds = 9
recall_ds = 3 # recall at 1
methods = set([x[0] for x in inlist])
for met in methods:
    data = [x[1:] for x in inlist if x[0] == met]

    # divide KB by 1000 to get MB
    for d in data:
        d[memory_ds] = float(int(d[memory_ds])/1000)
    
    # prepare the plot
    fig = plt.figure()
    plt.suptitle('Performance of ' + met)
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.85)
    par1 = host.twinx() # parasite axis on right

    # set limits on axes
    host.set_ylim(0.99 * float(min([float(d[time_ds]) for d in data])), 1.01 * float(max([float(d[time_ds]) for d in data])))
    par1.set_ylim(0.99 * float(min([float(d[memory_ds]) for d in data])), 1.01 * float(max([float(d[memory_ds]) for d in data])))

    parameters = [d[recall_ds] for d in data]
    p1 ,= host.plot(parameters, [d[time_ds] for d in data], 'ro', label = 'Search time [s]')
    p2 ,= par1.plot(parameters, [d[memory_ds] for d in data], 'bs', label = 'Max. memory usage [MB]')

    host.set_xlabel("Recall")
    host.set_ylabel("Search time [s]")
    par1.set_ylabel("Max. memory usage [MB]")

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    fig.savefig('plot' + str(met) + '.svg')
