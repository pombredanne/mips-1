# this script requires matplotlib and GNU time

import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
bin_names = ['./a.out', './b.out', './a.out']
args = ['5000', '10000', '20000', '30000', '60000']

num = 0
for binary in bin_names:
    num += 1
    data_series = [[], [], []]
    for a in args:
        time_result = subprocess.run(['/usr/bin/time', '-v', binary, a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        grep_result = subprocess.run(['grep', '-E', 'Output|Maximum resident'], input=time_result, stdout=subprocess.PIPE).stdout
        awk_result = subprocess.run(['awk', '{print $NF}'], input=grep_result, stdout=subprocess.PIPE).stdout
        decoded_result = awk_result.decode('utf-8').splitlines()
        for i in range(0,3):
            data_series[i].append(decoded_result[i])

    # prepare the plot
    fig = plt.figure()
    plt.suptitle('Performance of ' + binary)
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.7) # the plot takes only 70% of space to the left
    par1 = host.twinx() # parasite axis on right

    # parasite axis on right with offset to the right
    par2 = host.twinx()
    offset = 80
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset, 0))
    par2.axis["right"].toggle(all=True)

    # set limits on axes
    host.set_ylim(20, 1.15 * float(max([int(i) for i in data_series[0]])))
    par1.set_ylim(10, 1.1  * float(max([int(i) for i in data_series[1]])))
    par2.set_ylim( 0, 1.05 * float(max([int(i) for i in data_series[2]])))

    p1, = host.plot(args, data_series[0], 'ro', label="Execution time")
    p2, = par1.plot(args, data_series[1], 'bs', label="Precision")
    p3, = par2.plot(args, data_series[2], 'gx', label="Max. memory usage")

    host.set_xlabel("Parameter")
    host.set_ylabel("Execution time")
    par1.set_ylabel("Precision")
    par2.set_ylabel("Max. memory usage")

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    host.legend()
    fig.savefig('plot' + str(num) + '.svg')
