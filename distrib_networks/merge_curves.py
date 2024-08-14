import csv
import matplotlib.pyplot as plt
from math import ceil
import os
import numpy as np

def select_plot_color(network_name):
    if network_name == 'ehv1':
        color = 'red'
    elif network_name == 'ehv2':
        color = 'blue'
    elif network_name == 'ehv3':
        color = 'green'
    elif network_name == 'evh5':
        color = 'black'
    elif network_name == 'ehv6':
        color = 'orange'
    else:
        color = None
    return color

"""
# Old code with a single fault
def plot_power(curve_paths, network_names, fig_name):
    fig, axes = plt.subplots(ncols=2, figsize=(12,8))
    for curve_path, network_name in zip(curve_paths, network_names):
        t = []
        P = []
        Q = []
        with open(curve_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            reader.__next__()  # Skip header
            row = reader.__next__()
            P_init = float(row[1])
            Q_init = float(row[2])
            t.append(float(row[0]))
            P.append(float(row[1]))
            Q.append(float(row[2]))
            for row in reader:
                t.append(float(row[0]))
                P.append(float(row[1]))
                Q.append(float(row[2]))

        P = [y / 100 for y in P]
        Q = [y / 100 for y in Q]
        color = select_plot_color(network_name)
        # axes[0].plot(t, P, label=network_name, color=plt.cm.hsv(int(float(network_name[-1])*256.0/6.0)))
        axes[0].plot(t, P, label=network_name, color=color, linestyle='dotted')
        axes[1].plot(t, Q, label=network_name, color=color, linestyle='dotted')
    # axes[0].legend()
    # axes[1].legend()
    # Remove duplicate legend entries (https://stackoverflow.com/a/13589144)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys())
    axes[1].legend(by_label.values(), by_label.keys())

    axes[0].set_title('Active power (pu of total gross active load)')
    axes[1].set_title('Reactive power (pu of total gross active load)')
    plt.savefig(fig_name)
    plt.close()
"""

def plot_power(curve_paths, network_names, fig_name):
    selected_plot = 2

    nb_plots = nb_faults = len(curve_paths[0])

    nb_plots = 1

    ncols = ceil(nb_plots**0.5)
    nrows = ceil(nb_plots / ncols)
    fig_P, axes_P = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4,4))
    fig_Q, axes_Q = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4,4))


    t_selected = {}
    P_selected = {}
    Q_selected = {}

    for curve_path_, network_name in zip(curve_paths, network_names):
        for i, curve_path in enumerate(curve_path_):
            if i != selected_plot:
                continue
            i = 0

            t = []
            P = []
            Q = []
            with open(curve_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                reader.__next__()  # Skip header

                for row in reader:
                    t.append(float(row[0]))
                    Q_comp = float(row[3]) * 100
                    if network_name == 'ehv5':
                        nb_tfos = 3
                    else:
                        nb_tfos = 2
                    P_tfo = float(row[4]) * 100 * nb_tfos
                    Q_tfo = float(row[5]) * 100 * nb_tfos
                    # P.append(float(row[1]))
                    # Q.append(float(row[2]))
                    P.append(P_tfo)
                    Q.append(Q_tfo + Q_comp)


            P = [y / 100 for y in P]
            Q = [y / 100 for y in Q]
            color = select_plot_color(network_name)
            if network_name == 'ehv1':
                name = 'Small rural'
            elif network_name == 'ehv2':
                name = 'Large rural'
            elif network_name == 'ehv3':
                name = 'Semiurban'
            elif network_name == 'ehv5':
                name = 'Small urban'
            elif network_name == 'ehv6':
                name = 'Large urban'
            else:
                raise NotImplementedError()
            axes_P.plot(t, P, label=name, color=color, linestyle='dotted')
            axes_Q.plot(t, Q, label=name, color=color, linestyle='dotted')
            axes_P.set_xlim([0.8,2])
            axes_Q.set_xlim([0.8,2])
            axes_P.set_xlabel('Time [s]')
            axes_Q.set_xlabel('Time [s]')
            axes_P.set_ylabel('Active power [pu]')
            axes_Q.set_ylabel('Reactive power [pu]')

            if i == selected_plot:
                t_selected[network_name] = t
                P_selected[network_name] = P
                Q_selected[network_name] = Q
    # axes[0].legend()
    # axes[1].legend()
    # Remove duplicate legend entries (https://stackoverflow.com/a/13589144)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_P.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.95,0.976))
    fig_Q.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.95,0.43))

    # fig_P.suptitle('Active power (pu of total gross active load)')
    # fig_Q.suptitle('Reactive power (pu of total gross active load)')
    fig_name, ext = os.path.splitext(fig_name)
    fig_P.tight_layout()
    fig_Q.tight_layout()
    fig_P.savefig(fig_name + '_P' + ext)
    fig_Q.savefig(fig_name + '_Q' + ext)
    plt.close()

    """ for network_name in t_selected:
        with open('Equiv_{}.txt'.format(network_name), 'w') as out_file:
            writer = csv.writer(out_file, delimiter=' ')
            writer.writerow(['t', 'P', 'Q'])
            for t in range(len(t_selected[i])):
                row = [t_selected[i][t], P_selected[i][t], Q_selected[i][t]]
                writer.writerow(row) """


def plot_voltages(curve_paths, network_names, fig_name):
    nb_plots = nb_faults = len(curve_paths[0])
    ncols = ceil(nb_plots**0.5)
    nrows = ceil(nb_plots / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12,8))
    surplus_axes = int((ncols*nrows-nb_plots))
    for i in range(surplus_axes):
        fig.delaxes(axes[-1,-(1+i)])

    for curve_path_, network_name in zip(curve_paths, network_names):
        if network_name != 'ehv3':
            continue
        for i, curve_path in enumerate(curve_path_):
            t = []
            with open(curve_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                reader.__next__()  # Skip header
            V_max = []
            V_min = []
            all_V = []
            GSP_V = []
            with open(curve_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                header = reader.__next__()

                voltage_columns = []
                for j in range(len(header)):
                    if header[j][-9:] == 'Upu_value' and header[j][:12] == 'NETWORK_B-11':
                        voltage_columns.append(j)
                    if header[j] == 'NETWORK_B-99_Upu_value' or header[j] == 'NETWORK_B-100_Upu_value':
                        GSP_index = j

                for row in reader:
                    t.append(float(row[0]))
                    V = [float(row[j]) for j in voltage_columns]
                    V_max.append(max(V))
                    V_min.append(min(V))
                    all_V.append(V)
                    GSP_V.append(float(row[GSP_index]))

                color = select_plot_color(network_name)
                # axes[divmod(i, ncols)].set_ylim(bottom=0.0, top=1.15)
                axes[divmod(i, ncols)].set_xlim(left=0.9, right=2)
                axes[divmod(i, ncols)].plot(t, V_min, label='Min 11kV voltage', linestyle='dotted')
                axes[divmod(i, ncols)].plot(t, V_max, label='Max 11kV voltage', linestyle='dotted')
                axes[divmod(i, ncols)].plot(t, GSP_V, label='GSP voltage', alpha=1, linestyle='dashed')
                # axes[divmod(i, ncols)].set_title(network_name)
                # Remove duplicate legend entries (https://stackoverflow.com/a/13589144)
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                fig.legend(by_label.values(), by_label.keys(), 'lower right')

            if t[-1] < 4.5:
                raise RuntimeError('Dynawo failed the simulation', network_name)

    fig.tight_layout()
    plt.setp(axes[-1, :], xlabel='Time [s]')
    plt.setp(axes[:, 0], ylabel='Voltage [pu]')
    plt.savefig(fig_name)
    plt.close()


"""
def plot_voltages(curve_paths, network_names, fig_name):
    unique_network_names = sorted(list(set(network_names)))
    nb_plots = len(unique_network_names)
    # nb_plots = len(curve_paths)
    ncols = ceil(nb_plots**0.5)
    nrows = ceil(nb_plots / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12,8))
    surplus_axes = int((ncols*nrows-nb_plots))
    for i in range(surplus_axes):
        fig.delaxes(axes[-1,-(1+i)])
    for curve_path, network_name in zip(curve_paths, network_names):
        t = []
        V_max = []
        V_min = []
        all_V = []
        GSP_V = []
        with open(curve_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            header = reader.__next__()

            voltage_columns = []
            for i in range(len(header)):
                if header[i][-9:] == 'Upu_value' and header[i][:12] == 'NETWORK_B-11':
                    voltage_columns.append(i)
                if header[i] == 'NETWORK_B-99_Upu_value' or header[i] == 'NETWORK_B-100_Upu_value':
                    GSP_index = i

            for row in reader:
                t.append(float(row[0]))
                V = [float(row[i]) for i in voltage_columns]
                V_max.append(max(V))
                V_min.append(min(V))
                all_V.append(V)
                GSP_V.append(float(row[GSP_index]))

            plot_index = unique_network_names.index(network_name)
            axes[divmod(plot_index, ncols)].set_xlim([0.5,1.5])
            axes[divmod(plot_index, ncols)].plot(t, all_V, color='grey', alpha=0.2, linestyle='dotted')
            axes[divmod(plot_index, ncols)].plot(t, V_min, color='red', label='Min 11kV voltage', linestyle='dotted')
            axes[divmod(plot_index, ncols)].plot(t, V_max, color='blue', label='Max 11kV voltage', linestyle='dotted')
            axes[divmod(plot_index, ncols)].plot(t, GSP_V, color='black', label='GSP voltage', alpha=1, linestyle='dashed')
            # axes[divmod(plot_index, ncols)].set_title('Min and max voltages at the 11kV level')
            axes[divmod(plot_index, ncols)].set_title(network_name)
            # Remove duplicate legend entries (https://stackoverflow.com/a/13589144)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            # axes[0].legend(by_label.values(), by_label.keys())
            axes[divmod(plot_index, ncols)].legend(by_label.values(), by_label.keys())

            i = np.searchsorted(t, 1.1, side="right")
            print(max((np.array(V_min)-np.array(GSP_V))[i], (np.array(V_min)-np.array(GSP_V))[i+1]))
            print(max((np.array(V_max)-np.array(GSP_V))[i], (np.array(V_max)-np.array(GSP_V))[i+1]))
            print()

        if t[-1] < 4.5:
            raise RuntimeError('Dynawo failed the simulation', network_name)

    plt.setp(axes[-1, :], xlabel='Time [s]')
    plt.setp(axes[:, 0], ylabel='Voltage [pu]')
    plt.savefig(fig_name)
    plt.close()
"""