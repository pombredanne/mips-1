import sys
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

if len(sys.argv) < 2:
    print('Sorry, I don\'t know what to draw.')

with open('output.txt', 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    inlist = list(reader)

# ds = data series
time_ds = 8
recall_ds = 6 # recall at 10
inter_ds = 9 # intersection
number_of_queries = 100

minimum_time = float(min([float(d[time_ds+1]) for d in inlist]))
maximum_time = float(max([float(d[time_ds+1]) for d in inlist]))
minimum_recall = float(min([float(d[recall_ds+1]) for d in inlist]))
maximum_recall = float(max([float(d[recall_ds+1]) for d in inlist]))
minimum_inter = float(min([float(d[inter_ds+1]) for d in inlist]))
maximum_inter = float(max([float(d[inter_ds+1]) for d in inlist]))

data = []
for met in ['ivf', 'kmeans', 'quant', 'alsh']:
    data.append([x[1:] for x in inlist if x[0] == met])

# prepare the plot
fig = plt.figure()
host = host_subplot(111)
box = host.get_position()
host.set_position([box.x0, box.y0 + box.height*0.35, box.width, box.height * 0.7])
host.set_xscale('log')
host.set_xlabel("Time per query [ms]")
m_size=1.6

# set limits on axes
host.set_xlim(0.99 * 1000 * minimum_time/number_of_queries,
              1.01 * 1000 * maximum_time/number_of_queries)

if sys.argv[1] == 'recall':
    plt.suptitle('Recall vs. time')
    labels = ['Recall@10 - ivf', 'Recall@10 - kmeans',
              'Recall@10 - quant', 'Recall@10 - alsh']
    markers = ['C1o', 'C3o', 'C5o', 'C7o']
    host.set_ylabel("Recall@10")
    for i in [0, 1, 2, 3]:
        host.plot([float(d[time_ds])*1000/number_of_queries for d in data[i]],
                 [d[recall_ds] for d in data[i]],
                 markers[i], label = labels[i], markersize=m_size)
    host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig('plot_recall.pdf')

if sys.argv[1] == 'inter':
    plt.suptitle('Top-100 intersection vs. time')
    labels = ['Top-100 intersection - ivf', 'Top-100 intersection - kmeans',
              'Top-100 intersection - quant', 'Top-100 intersection - alsh']
    markers = ['C2s', 'C4s', 'C6s', 'C8s']
    host.set_ylabel("Top-100 intersection")
    for i in [0, 1, 2, 3]:
        host.plot([float(d[time_ds])*1000/number_of_queries for d in data[i]],
                 [d[inter_ds] for d in data[i]],
                 markers[i], label = labels[i], markersize=m_size)
    host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig('plot_intersection.pdf')
