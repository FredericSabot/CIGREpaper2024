import csv
import random
import argparse
import os
from math import sqrt, ceil
import subprocess
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from joblib import Parallel, delayed
from pathlib import Path
import pypowsybl as pp
import pandas as pd
import glob
import shutil

import RandomParameters
from RandomParameters import DynamicParameter, DynamicParameterList, StaticParameter, StaticParameterList
import MergeRandomOutputs
from ukgds_to_iidm import build_network_and_simulate
import ukgds_to_iidm

DYNAWO_ALGO_PATH = '/home/fsabot/Desktop/dynawo-algorithms/myEnvDynawoAlgorithms.sh'

def normaliseParameters(parameters, bounds):
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    norm_parameters = []
    for i in range(len(parameters)):
        if diff[i] == 0:
            norm_parameters.append(0.5)
        else:
            norm_parameters.append((parameters[i] - min_b[i]) / diff[i])
    return norm_parameters

def denormaliseParameters(normalised_parameters, bounds):
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    return min_b + normalised_parameters * diff

def de(fobj, bounds, mut=0.8, crossp=0.95, popsize=20, its=1000, init=None):
    """
    Differential Evolution (DE) algorithm, adapted from https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
    """
    global run_id
    run_id = 0  # Reset run_id
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    pop_denorm = denormaliseParameters(pop, bounds)

    if init is not None:
        pop_denorm[0] = init
        pop_denorm[-1] = init
    objs = [fobj(ind) for ind in pop_denorm]
    fitness = np.asarray([obj.total_obj for obj in objs])
    best_idx = np.argmin(fitness)
    best_obj = objs[best_idx]
    best = pop_denorm[best_idx]
    best_it = 0
    best_converged = False
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = denormaliseParameters(trial, bounds)
            obj = fobj(trial_denorm)
            f = obj.total_obj
            converged = obj.converged
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best_obj = obj
                    best = trial_denorm
                    best_it = i
                    best_converged = converged
        yield DEResults(best_idx, best_it, best, best_obj, best_converged)
        if best_converged:
            return

@dataclass
class Objective:
    total_obj: float
    obj_name_disturb: np.ndarray
    parameter_dist: list
    converged: bool

@dataclass
class DEResults:
    best_index: int
    best_iteration: int
    best_parameters: list
    best_obj: Objective
    converged: bool

    def __repr__(self) -> str:
        return str(self.best_index) + ',' + str(self.best_iteration) + ',' + str(self.best_parameters) + ',' + str(self.best_obj) + ',' + str(self.converged)

def refineStaticParameters(static_bounds):
    pass
    return static_bounds

class DynamicParameterListWithBounds:
    def __init__(self, bounds_csv):
        self.param_list = DynamicParameterList()
        self.bounds = []

        with open(bounds_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['ParamSet_id', 'Param_id', 'L_bound', 'U_bound']:
                raise Exception("Incorrect format of %s" % bounds_csv)

            for row in spamreader:
                paramSetId = row[0]
                paramId = row[1]
                bounds = [float(row[2]), float(row[3])]

                self.param_list.append(DynamicParameter(paramSetId, paramId, None))
                self.bounds.append(bounds)

    def __append__(self, parameter, bounds):
        """
        @param parameter DynamicParameter object to append
        @param bounds tuple (min_bound, max_bound)
        """
        self.param_list.append(parameter)
        self.bounds.append(bounds)

    def valueListToParameterList(self, value_list):
        if len(value_list) != len(self.bounds):
            raise ValueError('value_list and self_bounds should have the same length')

        parameter_list = DynamicParameterList()
        for i in range(len(value_list)):
            parameter_list.append(self.param_list[i])
            parameter_list[i].value = value_list[i]
        return parameter_list


class StaticParameterListWithBounds:
    def __init__(self, bounds_csv):
        self.param_list = StaticParameterList()
        self.bounds = []

        with open(bounds_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['Component_type', 'Component_name', 'Param_id', 'L_bound', 'U_bound']:
                print(row)
                raise Exception("Incorrect format of %s" % bounds_csv)

            for row in spamreader:
                componentType = row[0]
                componentName = row[1]
                paramId = row[2]
                bounds = [float(row[3]), float(row[4])]

                self.param_list.append(StaticParameter(componentType, componentName, paramId, None))
                self.bounds.append(bounds)

    def __append__(self, parameter, bounds):
        """
        @param parameter DynamicParameter object to append
        @param bounds tuple (min_bound, max_bound)
        """
        self.param_list.append(parameter)
        self.bounds.append(bounds)

    def valueListToParameterList(self, value_list):
        if len(value_list) != len(self.bounds):
            raise ValueError('value_list and self_bounds should have the same length')

        parameter_list = StaticParameterList()
        for i in range(len(value_list)):
            parameter_list.append(self.param_list[i])
            parameter_list[-1].value = value_list[i]
        return parameter_list


def runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_P, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None, disturbance_ids=None, nb_threads='5'):
    RandomParameters.writeParametricSAInputs(working_dir, fic_MULTIPLE, network_name, output_dir_name, static_parameters, dyn_parameters,
            run_id, target_P, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids)

    output_dir = os.path.join(working_dir, output_dir_name)
    cmd = [DYNAWO_ALGO_PATH, 'SA', '--directory', output_dir, '--input', 'fic_MULTIPLE.xml',
            '--output' , 'aggregatedResults.xml', '--nbThreads', nb_threads]
    output = subprocess.run(cmd, capture_output=True, text=True)
    if output.stderr != '':
        print(output.stderr, end='')


def runRandomSA(static_csv, dynamic_csv, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None):
    full_network_name = os.path.join(working_dir, network_name)
    static_parameters = RandomParameters.randomiseStaticParams(full_network_name + '.iidm', static_csv)
    dyn_parameters = RandomParameters.randomiseDynamicParams(full_network_name + '.par', dynamic_csv)

    runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type)


def runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_P, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None, total_load=None, total_gen=None, disturbance_ids=None):
    dyn_value_list, static_value_list = value_list[:nb_dyn_params], value_list[nb_dyn_params:]
    dyn_parameters = dyn_bounds.valueListToParameterList(dyn_value_list)
    static_parameters = static_bounds.valueListToParameterList(static_value_list)

    if total_load is not None and total_gen is not None:
        static_parameters = refineStaticParameters(static_parameters, total_load, total_gen)

    runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_P, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids)


def objective(fitted_curves, random_curves, target_percentile, value_list, bounds, lasso_factor, parameter_names, convergence_criteria = 1):
    median = np.median(random_curves, axis=2)  # average over runs
    sigma = np.std(random_curves, axis=2, ddof=1)
    percentile = np.percentile(random_curves, axis=2, q=target_percentile)
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(max(value, 5e0), 1e-2 * abs(median[x,y,z]))  # Avoid division by 0 + allow some tolerance
        # sigma[x,y,z] = min(sigma[x,y,z], abs(0.1 * max(median[x,y,0], median[x,y,z])))  # Limit tolerance when very high dispersion # Issue if median close to 0 at some point

    obj = ((fitted_curves - percentile) / sigma)
    for (x,y,z), value in np.ndenumerate(obj):
        if target_percentile > 50:
            if value > 0:
                obj[x, y, z] = value / 2
        elif target_percentile < 50:
            if value < 0:
                obj[x, y, z] = value / 2
    obj = obj**2
    obj = np.clip(obj, 0, 50)  # Allow for large errors if they only occur during a very short period.

    obj_name_disturb = np.mean(obj, axis=2)  # average over time
    if np.max(obj_name_disturb) <= convergence_criteria:
        converged = True
    else:
        converged = False
    obj = np.sum(obj_name_disturb, axis=0)  # sum over curve_names (typically P and Q at point of common coupling)
    obj = np.mean(obj)  # average over disturbances

    normalised_parameters = normaliseParameters(value_list, bounds)
    parameter_dist = [abs(i - 0.5) for i in normalised_parameters]
    total_obj = obj + lasso_factor * sum([0 if isTrippingParameter(parameter_names[i]) else parameter_dist[i] for i in range(len(parameter_dist))])

    return Objective(total_obj, obj_name_disturb, parameter_dist, converged)

def plotCurves(curves, plot_name, fitted_curves = None, target_percentile=None, time_precision=1e-2):  # ndarray(curve_name, scenario, run, t_step)
    nb_disturb = curves.shape[1]
    nb_runs = curves.shape[2]
    nb_time_steps = curves.shape[3]
    t_axis = np.array([i * time_precision for i in range(nb_time_steps)])
    sqrt_d = int(ceil(sqrt(nb_disturb)))
    rcParams['figure.figsize'] = 12, 7.2
    curve_names = ['P (MW)', 'Q (MVar)']

    median = np.median(curves, axis=2)  # average over runs
    sigma = np.std(curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)
    percentile_5, percentile_95 = np.percentile(curves, axis=2, q=[5, 95])
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(max(value, 2e0), 1e-2 * abs(median[x,y,z]))  # Avoid division by 0 + allow some tolerance

    if fitted_curves is not None and target_percentile is not None:
        percentile = np.percentile(curves, axis=2, q=target_percentile)
        obj = ((fitted_curves - percentile) / sigma)
        for (x,y,z), value in np.ndenumerate(obj):
            if value < 0:
                obj[x, y, z] = value / 2
        obj = obj**2
        obj = np.clip(obj, 0, 50)  # Allow for large errors if they only occur during a very short period
    else:
        obj = None

    for c in range(curves.shape[0]):
        fig, axs = plt.subplots(sqrt_d, sqrt_d)

        surplus_axes = int((sqrt_d*sqrt_d-curves.shape[1]))
        for i in range(surplus_axes):
            fig.delaxes(axs[-1,-(1+i)])

        for d in range(curves.shape[1]):
            axs2 = axs[d//sqrt_d, d%sqrt_d].twinx()
            axs[d//sqrt_d, d%sqrt_d].set_title('Disturbance %d' % (d + 1))
            axs[d//sqrt_d, d%sqrt_d].set_xlabel('Time (s)')
            axs[d//sqrt_d, d%sqrt_d].set_ylabel(curve_names[c])
            axs2.set_ylabel('Error')
            axs2.set_ylim([0, 10])
            for r in range(nb_runs):
                axs[d//sqrt_d, d%sqrt_d].plot(t_axis, curves[c,d,r,:], ':', linewidth=1, alpha=0.3)
            axs[d//sqrt_d, d%sqrt_d].plot(t_axis, percentile_5[c,d,:], label='5th percentile', zorder=1000, alpha=0.7)
            axs[d//sqrt_d, d%sqrt_d].plot(t_axis, percentile_95[c,d,:], label='95th percentile', zorder=1000, alpha=0.7)
            # axs[d//sqrt_d, d%sqrt_d].plot(t_axis, median[c,d,:], 'green', label='Median', zorder=1000, alpha=0.7)

            if obj is not None:
                axs2.plot(t_axis, obj[c,d,:], label='Error', zorder=1000, alpha=0.3)

            if fitted_curves is not None:
                axs[d//sqrt_d, d%sqrt_d].plot(t_axis, fitted_curves[c,d,:], 'red', label='Fit', zorder=3000)
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')
        plt.tight_layout()
        plt.savefig(plot_name + '_%d.pdf' % c, bbox_inches='tight')
        plt.close()

def getParameterNames(dyn_bounds, static_bounds):
    parameter_names = []
    for parameter in dyn_bounds.param_list:
        parameter_names.append(parameter.set_id + '_' + parameter.id)
    for parameter in static_bounds.param_list:
        parameter_names.append(parameter.component_name + '_' + parameter.id)
    return parameter_names

def printParameters(de_results : DEResults, dyn_bounds, static_bounds, output_file = None, additional_parameters = None):
    parameter_names = getParameterNames(dyn_bounds, static_bounds)

    parameter_values = de_results.best_parameters
    for i in range(len(parameter_values)):
        print(parameter_names[i], parameter_values[i])
    if additional_parameters is not None:
        for additional_parameter in additional_parameters:
            print(additional_parameter[0], additional_parameter[1])

    if output_file is not None:
        with open(output_file, 'w') as file:
            parameter_values = de_results.best_parameters
            for i in range(len(parameter_values)):
                file.write(parameter_names[i] + ' ' + str(parameter_values[i]))
                file.write('\n')
            if additional_parameters is not None:
                for additional_parameter in additional_parameters:
                    file.write(additional_parameter[0] + ' ' + str(additional_parameter[1]))
                    file.write('\n')


def isTrippingParameter(parameter_name):
    if 'LVRT' in parameter_name:
        return True
    else:
        return False


def build_dynamic_equivalent(load_ratio, der_capacity_factor, der_installed_share, der_legacy_share, target_percentile):
    random.seed(1)
    start_time = time.time()

    parameter_string = '_'.join([str(i) for i in [load_ratio, der_capacity_factor, der_installed_share, der_legacy_share]])

    working_dir = 'dynawo_files/Equivalent/'
    reduced_fic_MULTIPLE = os.path.join(working_dir, 'reduced_fic.xml')
    reduced_network_name = 'reduced'
    csv_par_bounds = os.path.join(working_dir, 'params_dyd_bounds.csv')
    csv_iidm_bounds = os.path.join(working_dir, 'params_iidm_bounds.csv')
    target_Q = 0
    nb_runs_random = 3  # Only consider sensitivity case on motor share
    slack_load_id = 'LOAD-slack'
    slack_gen_id = 'GEN-slack'
    slack_gen_type = "Generator"
    curve_names = ['GEN-slack_generator_PGen', 'GEN-slack_generator_QGen']
    time_precision = 0.01

    rerun_random = False
    rerun_de = True

    if rerun_random:  # If intermediary results changes, need to rerun following steps
        rerun_de = True

    ###
    # Part 1: random runs
    ###

    Path('RandomCurves').mkdir(parents=True, exist_ok=True)
    random_curves_save = os.path.join('RandomCurves', parameter_string + '.npy')

    seeds = range(nb_runs_random)
    # seeds = [None]
    network_names = ['ehv{}'.format(i) for i in range(1, 7) if i != 4]

    if Path(random_curves_save).exists() and not rerun_random:
        random_curves = np.load(random_curves_save)
    else:
        PARALLEL = False
        if PARALLEL:
            results = Parallel(n_jobs=5)(delayed(build_network_and_simulate)(network_name, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, seed, True) for network_name in network_names for seed in seeds)
        else:
            results = []
            for network_name in network_names:
                for seed in seeds:
                    results.append(build_network_and_simulate(network_name, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, seed, True))

        output_curve_paths = []
        network_names = []
        run_fic_MULTIPLE = []
        for result in results:
            output_curve_paths.append(result[0])
            network_names.append(result[1])
            run_fic_MULTIPLE.append(result[2])  # Different networks are interpreted as independent random runs for the purpose of equivalencing

        random_curves = MergeRandomOutputs.mergeCurvesFromFics(run_fic_MULTIPLE, curve_names, time_precision)
        np.save(random_curves_save, random_curves)

    # Print some statistics
    sigma = np.std(random_curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0
    std_error = sigma / len(seeds) / len(network_names)
    print('Nb random runs: %d, Std error: %f' % (nb_runs_random * len(network_names), std_error.max()))

    randomising_time = time.time()

    Path('Random_Curves').mkdir(parents=True, exist_ok=True)
    plotCurves(random_curves, os.path.join('Random_Curves', 'Random_' + parameter_string))
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))

    ###
    # Part 2: optimisation
    ###

    der_legacy_share = max(der_legacy_share, 1e-3)  # At least some amount of legacy and non-legacy for numerical reasons
    der_legacy_share = min(der_legacy_share, 1-1e-3)

    equivalent = pp.network.load(os.path.join(working_dir, 'reduced.iidm'))
    """ current_equivalent_path = os.path.join(working_dir, 'current_equivalent')
    equivalent_files = glob.glob(os.path.join(working_dir, '*'))
    working_dir = os.path.join(working_dir, 'current_equivalent')
    for file in equivalent_files:
        shutil.copy(file, working_dir) """
    equivalent.update_loads(pd.DataFrame({'p0' : 100 * load_ratio}, index = ['LOAD']))
    equivalent.update_generators(pd.DataFrame({'max_p' : 100 * der_installed_share * der_legacy_share}, index = ['IBG-legacy']))
    equivalent.update_generators(pd.DataFrame({'target_p' : 100 * der_installed_share * der_capacity_factor * der_legacy_share}, index = ['IBG-legacy']))
    equivalent.update_generators(pd.DataFrame({'max_p' : 100 * der_installed_share * (1-der_legacy_share)}, index = ['IBG-G99']))
    equivalent.update_generators(pd.DataFrame({'target_p' : 100 * der_installed_share * der_capacity_factor * (1-der_legacy_share)}, index = ['IBG-G99']))
    ukgds_to_iidm.dump_network_file(equivalent, os.path.join(working_dir, 'reduced'))

    parameter_string += '_' + str(target_percentile)

    percentile = np.percentile(random_curves, axis=2, q=target_percentile)
    target_P = percentile[0, 0, 0]

    np.random.seed(int(42))

    dyn_bounds = DynamicParameterListWithBounds(csv_par_bounds)
    dyn_bounds_list = dyn_bounds.bounds
    nb_dyn_params = len(dyn_bounds_list)

    static_bounds = StaticParameterListWithBounds(csv_iidm_bounds)
    static_bounds_list = static_bounds.bounds

    bounds = dyn_bounds_list + static_bounds_list
    parameter_names = getParameterNames(dyn_bounds, static_bounds)

    def fobj2(value_list, lasso_factor, output_dir, disturbance_ids = None, convergence_criteria = 1.0, rerun = True):
        global run_id
        output_dir_name = os.path.join(output_dir, "It_%03d" % run_id)
        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        if rerun:
            runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, reduced_fic_MULTIPLE, reduced_network_name, run_id, target_P, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids=disturbance_ids)
        curves = MergeRandomOutputs.mergeCurvesFromFics([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        curves = curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)
        if disturbance_ids is not None:
            random_curves_considered = random_curves[:,disturbance_ids,:]
        else:
            random_curves_considered = random_curves
        obj = objective(curves, random_curves_considered, target_percentile, value_list, bounds, lasso_factor, parameter_names, convergence_criteria)
        print('Run id: %d, objective: %f' %(run_id, obj.total_obj))
        run_id += 1
        return obj

    # DE parameters
    pop_size = 10
    nb_max_iterations_DE = 20

    # DE
    convergence_criteria = 1 # max(1, np.max(results[-1].best_obj.obj_name_disturb))
    print('\nDE')
    def fobj(value_list):
        return fobj2(value_list, 0, 'Optimisation', rerun=rerun_de, convergence_criteria=convergence_criteria)
    results = list(de(fobj, bounds, popsize=pop_size, its=nb_max_iterations_DE))
    print('DE', ": objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))

    # Results
    best_run_id = results[-1].best_index + pop_size * (results[-1].best_iteration + 1)
    network = pp.network.load(os.path.join(working_dir, 'Optimisation', "It_%03d" % best_run_id, reduced_network_name + '.iidm'))
    tap = network.get_ratio_tap_changers().tap.values[0]
    reduced_loads = network.get_loads()
    slack_load_p = reduced_loads.at['LOAD-slack', 'p0']
    slack_load_q = reduced_loads.at['LOAD-slack', 'q0']

    Path('Optimised_Parameters').mkdir(parents=True, exist_ok=True)
    printParameters(results[-1], dyn_bounds, static_bounds, os.path.join('Optimised_Parameters', 'Optimised_parameters_' + parameter_string + '.txt'),
                    additional_parameters = [('tap', tap), ('slack_load_p', slack_load_p), ('slack_load_q', slack_load_q)])

    print("Objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))
    convergence_evolution = [r.best_obj.total_obj for r in results]
    plt.plot(convergence_evolution)
    plt.savefig('Convergence.png', bbox_inches='tight')
    plt.close()

    optimising_time = time.time()
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))
    print('Spent %.1fs on optimising the reduced model' % (optimising_time-randomising_time))


    fitted_curves = MergeRandomOutputs.mergeCurvesFromFics([os.path.join(working_dir, 'Optimisation', "It_%03d" % best_run_id, 'fic_MULTIPLE.xml')], curve_names, time_precision)
    fitted_curves = fitted_curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)

    Path('Fitted_Curves').mkdir(parents=True, exist_ok=True)
    plotCurves(random_curves, os.path.join('Fitted_Curves', 'Fit_' + parameter_string), fitted_curves, target_percentile)

    print('\nDE best run id:', best_run_id)
    print("Objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))

    return results[-1]


if __name__ == '__main__':
    NGET_loads = ['HARK', 'STEW', 'HAWP', 'NORT']
    SPT_loads = ['LOAN', 'WYHI', 'HUNE', 'STHA', 'COCK']
    SSEN_loads = ['TUMM', 'TEAL', 'KINT', 'PEHE', 'BEAU', 'BLHI']

    for scenario in range(5):
        if scenario == 0:
            year = 2021
            SCENARIO_NAME = 'Winter_{}_leading'.format(year)
            SCOTLAND_WIND_AVAILABILITY = 0.9  # Leads to 2.64GW through B6 (limit = 3.880)
            NGET_WIND_AVAILABILITY = 0.8
        elif scenario == 1:
            year = 2030
            SCENARIO_NAME = 'Winter_{}_leading'.format(year)
            SCOTLAND_WIND_AVAILABILITY = 0.9  # Leads to 3.35GW through B4 (limit 5.2GW), 2.5GW through B6 (limit 4GW) + 9.8GW through HVDCs
            NGET_WIND_AVAILABILITY = 0.8
        elif scenario == 2:
            year = 2021
            SCENARIO_NAME = 'SummerPM_{}_leading'.format(year)
            SCOTLAND_WIND_AVAILABILITY = 0.7  # Leads to 3.05GW through B4 (limit 3.3), 3.1 through G6 (limit 3.8) + 1.6GW HVDC, not dynamically stable, stable with lower B4 flow
            NGET_WIND_AVAILABILITY = 0.7
        elif scenario == 3:
            year = 2030
            SCENARIO_NAME = 'SummerPM_{}_leading'.format(year)
            SCOTLAND_WIND_AVAILABILITY = 0.7 # Leads to 2.2GW through B4 (limit 5.2), 3.0 through B4 (limit 4GW) + 9GW through HVDCs
            NGET_WIND_AVAILABILITY = 0.3
        elif scenario == 4:
            year = 2030
            SCENARIO_NAME = 'SummerAM_{}_leading'.format(year)
            SCOTLAND_WIND_AVAILABILITY = 0.1
            NGET_WIND_AVAILABILITY = 0.1

        if 'Winter' in SCENARIO_NAME:
            CHP_FACTOR = 0.7
            SOLAR_FACTOR = 0  # Winter evening
        elif 'SummerPM' in SCENARIO_NAME:
            CHP_FACTOR = 0.2
            SOLAR_FACTOR = 0.68  # All-time peak according to https://www.solar.sheffield.ac.uk/pvlive/ (checked on 14/12/2023, peak reached on 2023-04-20 12:30PM which is not really in the summer)
        elif 'SummerAM' in SCENARIO_NAME:
            CHP_FACTOR = 0.2
            SOLAR_FACTOR = 0.1

        SOLAR_CAPACITY_CORRECTION = 0.8  # The inverter of PV is often under-dimnensioned as PV rarely output 100% of their capacity
        # So, reduce the installed capacity of PV compared to FES data and increase the capacity factor to compensate (same total
        # output, but lower capacity)
        if SOLAR_FACTOR > SOLAR_CAPACITY_CORRECTION:
            raise ValueError()
        SOLAR_FACTOR = SOLAR_FACTOR / SOLAR_CAPACITY_CORRECTION

        peak_loads = {}
        with open(os.path.join('..', 'FES data', 'aggregated', 'Winter_{}_leading'.format(year) + '.csv')) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header
            for row in reader:
                load_name = row[0]
                P_gross = float(row[1])
                peak_loads[load_name] = P_gross
        with open(os.path.join('..', 'FES data', 'aggregated', 'SummerPM_{}_leading'.format(year) + '.csv')) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header
            for row in reader:
                load_name = row[0]
                P_gross = float(row[1])
                peak_loads[load_name] = max(P_gross, peak_loads[load_name])

        with open(os.path.join('..', 'FES data', 'aggregated', SCENARIO_NAME + '.csv')) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header
            for row in reader:
                load_name = row[0]
                P_gross = float(row[1])
                Q_net = float(row[2])
                # storage = 0  # Included in gross load
                solar = float(row[4]) * SOLAR_CAPACITY_CORRECTION
                wind = float(row[5])
                # hydro = 0  # Neglected
                other = float(row[7])

                if load_name == 'NGET':
                    continue
                elif load_name in NGET_loads:
                    wind_factor = NGET_WIND_AVAILABILITY
                else:
                    wind_factor = SCOTLAND_WIND_AVAILABILITY

                P_gross = P_gross - other * CHP_FACTOR

                load_ratio = P_gross / peak_loads[load_name]
                der_installed_share = (solar + wind) / peak_loads[load_name]
                der_capacity_factor = (solar * SOLAR_FACTOR + wind * wind_factor) / (solar + wind)

                if load_name in NGET_loads:
                    wind_non_grid_code_share = 1 - 0.29
                elif load_name in SPT_loads:
                    wind_non_grid_code_share = 1 - 0.43
                elif load_name in SSEN_loads:
                    wind_non_grid_code_share = 1 - 0.73
                else:
                    raise NotImplementedError(load_name)

                for case in ['default', 'G99_extended']:
                    if year == 2021:
                        der_legacy_share = (solar + wind * wind_non_grid_code_share) / (solar + wind)
                    elif year == 2030:
                        if case == 'default':  # Legacy counts both plants installed before Apr 2019 and plants with a capacity < 16A per phase
                            der_legacy_share = (solar * (0.23 + 0.41) + wind * 0.83 * wind_non_grid_code_share) / (solar + wind)
                        elif case == 'G99_extended':  # G99 also applied to new PV plants even if capa < 16A
                            der_legacy_share = (solar * 0.23 + wind * 0.83 * wind_non_grid_code_share) / (solar + wind)

                    if case == 'G99_extended':
                        if year != 2030:
                            continue

                    load_ratio = round(load_ratio, 2)
                    der_capacity_factor = round(der_capacity_factor, 2)
                    der_installed_share = round(der_installed_share, 2)
                    der_legacy_share = round(der_legacy_share, 2)

                    for target_percentile in [5, 95]:
                        parameter_string = '_'.join([str(i) for i in [load_ratio, der_capacity_factor, der_installed_share, der_legacy_share, target_percentile]])
                        print('Launching case: load_name', load_name, 'scenario', SCENARIO_NAME, 'load ratio', load_ratio, 'capacity factor', der_capacity_factor,
                                'installed share', der_installed_share, 'legacy', der_legacy_share, 'percentile', target_percentile)
                        if not Path(os.path.join('Optimised_Parameters', 'Optimised_parameters_' + parameter_string + '.txt')).exists():
                            results = build_dynamic_equivalent(load_ratio, der_capacity_factor, der_installed_share, der_legacy_share, target_percentile)
                            obj = results.best_obj.total_obj
                            if obj > 2:
                                raise RuntimeError('Non convergence for case', load_ratio, der_capacity_factor, der_installed_share, der_legacy_share, target_percentile,
                                                    '\nInvestigate and delete Optimised_parameters for this case')
                            print('Completed case: load_name', load_name, 'scenario', SCENARIO_NAME, 'load ratio', load_ratio, 'capacity factor', der_capacity_factor,
                                'installed share', der_installed_share, 'legacy', der_legacy_share, 'percentile', target_percentile,
                                ', objective:', obj)
                        else:
                            print('Case already done')