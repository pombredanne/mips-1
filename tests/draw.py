import argparse
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

MARKER_SIZE = 3.5
WIDTH     = 0.5
TIME_DS   = 8
RECALL_DS = 6  # recall at 10
INTER_DS  = 9  # intersection
N_QUERIES = 100


def plot_recall(fig, host, data, limits):
    labels = ['Recall@10 - ivf', 'Recall@10 - kmeans', 'Recall@10 - quant', 'Recall@10 - alsh']
    markers = ['ro', 'yo', 'bo', 'go']

    plt.suptitle('Recall vs. time')
    host.set_ylabel("Recall@10")
    host.set_ylim(0.9 * limits[0], 1.1 * limits[1])

    # TODO generate JS file for top-100 intersection just like below
    with open('script.js', 'w') as js_output:
        for i, alg in enumerate(['ivf', 'kmeans', 'quant', 'alsh']):
            js_output.write('var data_' + str(i) + ' = {\n x: ')
            x = [float(d[TIME_DS]) * 1000 / N_QUERIES for d in data[i]]
            y = [d[RECALL_DS] for d in data[i]]
            js_output.write(str(x))
            js_output.write(',\n y: ')
            js_output.write(str([float(a) for a in y]))
            if alg == 'ivf':
                text = ['nprobe=' + str(d[0]) for d in data[i]]
            elif alg == 'kmeans':
                text = ['layers=' + str(d[0]) + ' op_tr=' + str(d[4]) +
                        ' aug_type=' + str(d[1]) + ' U=' + str(d[3])
                        for d in data[i]]
            elif alg == 'quant':
                text = ['subsp=' + str(d[0]) + ' centr=' + str(d[1])
                        for d in data[i]]
            elif alg == 'alsh':
                text = ['tabl=' + str(d[0]) + ' fun=' + str(d[1]) +
                        ' aug_type=' + str(d[3]) + ' U=' + str(d[4]) +
                        ' r=' + str(d[2])
                        for d in data[i]]
            js_output.write(',\n text: ')
            js_output.write(str(text))
            js_output.write(',\n mode: \'markers\',\n')
            js_output.write('name :\'' + str(alg) + '\' };\n')

            host.plot(x, y, markers[i], label=labels[i],
                      markersize=MARKER_SIZE, mfc='none', markeredgewidth=WIDTH)

        js_output.write('\nvar data = [data_0, data_1, data_2, data_3];\n\n')
        js_output.write('var layout = { title:\'Recall vs. time\', yaxis: { title: \'Recall@10\' }, ')
        js_output.write('xaxis: { type: \'log\', autorange: true, title: \'Time [ms]\' }, height : 1000 };\n\n')
        js_output.write('Plotly.newPlot(\'myDiv\', data, layout);')

    host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig('plot_recall.pdf')


def plot_intersect(fig, host, data, limits):
    labels = ['Top-100 intersection - ivf', 'Top-100 intersection - kmeans',
              'Top-100 intersection - quant', 'Top-100 intersection - alsh']
    markers = ['rs', 'ys', 'bs', 'gs']

    plt.suptitle('Top-100 intersection vs. time')
    host.set_ylabel("Top-100 intersection")
    host.set_ylim(0.9 * limits[0], 1.1 * limits[1])

    for i in range(4):
        x = [float(d[TIME_DS]) * 1000 / N_QUERIES for d in data[i]]
        y = [d[INTER_DS] for d in data[i]]

        host.plot(x, y, markers[i], label=labels[i],
                  markersize=MARKER_SIZE, mfc='none', markeredgewidth=WIDTH)

    host.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig('plot_intersection.pdf')


def main(args):

    with open(args.input) as f:
        reader = csv.reader(f, delimiter="\t")
        inlist = list(reader)

    minimum_time   = min([float(d[TIME_DS + 1]) for d in inlist])
    maximum_time   = max([float(d[TIME_DS + 1]) for d in inlist])
    minimum_recall = min([float(d[RECALL_DS + 1]) for d in inlist])
    maximum_recall = max([float(d[RECALL_DS + 1]) for d in inlist])
    minimum_inter  = min([float(d[INTER_DS + 1]) for d in inlist])
    maximum_inter  = max([float(d[INTER_DS + 1]) for d in inlist])

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

    # set limits on axes
    host.set_xlim(0.9 * 1000 * minimum_time / N_QUERIES,
                  1.1 * 1000 * maximum_time / N_QUERIES)

    if args.mode == 'recall':
        plot_recall(fig=fig, host=host, data=data, limits=(minimum_recall, maximum_recall))

    if args.mode == 'inter':
        plot_intersect(fig=fig, host=host, data=data, limits=(minimum_inter, maximum_inter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("A simple utility to create plots with benchmark results.")
    parser.add_argument("--input", default="output.txt",
                        help="File from which the data should be read.")
    parser.add_argument("--mode", choices={"recall", "inter"}, default="recall",
                        help="Controls what plot will be drawn")
    args = parser.parse_args()

    main(args)
