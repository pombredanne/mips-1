import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

with open('output.txt', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    inlist = list(reader)

# ds = data series
time_ds = 6
#memory_ds = 9
recall_ds = 4 # recall at 1
number_of_queries = 100
"""
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

"""

minimum_time = float(min([float(d[time_ds+1]) for d in inlist]))
maximum_time = float(max([float(d[time_ds+1]) for d in inlist]))
#minimum_mem = float(min([float(d[memory_ds+1]) for d in inlist]))
#maximum_mem = float(max([float(d[memory_ds+1]) for d in inlist]))
minimum_recall = float(min([float(d[recall_ds+1]) for d in inlist]))

data = []
for met in ['ivf', 'kmeans', 'quant']:
    data.append([x[1:] for x in inlist if x[0] == met])

# divide KB by 1000 to get MB
#for i in data:
#    for d in i:
#        d[len(d) - 1] = float(int(d[len(d) - 1])/1000)
#minimum_mem /= 1000
#maximum_mem /= 1000

# prepare the plot
fig = plt.figure()
plt.suptitle('Performance of algorithms')
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.85)
#par1 = host.twinx() # parasite axis on right

box = host.get_position()
host.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.85])

# set limits on axes
host.set_ylim(0.99 * 1000 * minimum_time/number_of_queries, 1.01 * 1000 * maximum_time/number_of_queries)
#par1.set_ylim(0.99 * minimum_mem, 1.01 * maximum_mem)
host.set_xlim(minimum_recall, 1.005)
#par1.set_ylim(0, 1.01 * maximum_mem)

label_time = ['Search time - ivf', 'Search time - kmeans', 'Search time - quant']
#label_mem = ['Memory usage - ivf', 'Memory usage - kmeans', 'Memory usage - quant']
marker_time = ['C1o', 'C3o', 'C5o']
#marker_mem = ['C2s', 'C4s', 'C6s']

for i in [0, 1, 2]:
    host.plot([d[recall_ds] for d in data[i]], [float(d[time_ds])*1000/number_of_queries   for d in data[i]], marker_time[i], label = label_time[i], markersize=1.5)
    #par1.plot([d[recall_ds] for d in data[i]], [d[memory_ds] for d in data[i]], marker_mem[i], label = label_mem[i], arkersize=2.8)

host.set_xlabel("Recall@10")
host.set_ylabel("Time per query [ms]")
#par1.set_ylabel("Max. memory usage [MB]")

host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
fig.savefig('plot.svg')

