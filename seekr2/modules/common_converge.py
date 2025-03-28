"""
common_converge.py

Contain functions used by converge.py script to determine the
convergence of SEEKR2 calculations.
"""

import os
import re
import glob
import math
from collections import defaultdict
import functools
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
from parmed import unit

import seekr2.analyze as analyze
import seekr2.modules.common_base as base
import seekr2.modules.common_analyze as common_analyze
import seekr2.modules.mmvt_analyze as mmvt_analyze
import seekr2.modules.elber_analyze as elber_analyze

# The default number of points to include in convergence plots
DEFAULT_NUM_POINTS = 100

# How many steps to skip before computing the convergence values
DEFAULT_SKIP = 0

# The threshold beneath which to skip plotting the convergence
MIN_PLOT_NORM = 1e-18

# The interval between which to update the user on convergence script progress
PROGRESS_UPDATE_INTERVAL = DEFAULT_NUM_POINTS // 10

def get_bd_transition_counts(model):
    """
    Obtain how many transitions have occurred in the BD stage.
    """
    assert model.using_bd(), "No valid BD program settings provided."
    output_file_glob = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory, 
        model.k_on_info.bd_output_glob)
    output_file_list = glob.glob(output_file_glob)
    output_file_list = base.order_files_numerically(output_file_list)
    compute_rate_constant_program = os.path.join(
        model.browndye_settings.browndye_bin_dir, "compute_rate_constant")
    bd_transition_counts = {}
    if len(output_file_list) > 0:
        k_ons_src, k_on_errors_src, reaction_probabilities, \
            reaction_probability_errors, transition_counts = \
            common_analyze.browndye_run_compute_rate_constant(
                compute_rate_constant_program, output_file_list, 
                sample_error_from_normal=False)
        bd_transition_counts["b_surface"] = transition_counts
    return bd_transition_counts

def analyze_bd_only(model, data_sample):
    """
    If there are missing MD statistics, then perhaps only a BD analysis
    should be performed. This function only performs a BD analysis on
    a particular data sample.
    """
    if model.k_on_info is None:
        return
    output_file_glob = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory, 
        model.k_on_info.bd_output_glob)
    output_file_list = glob.glob(output_file_glob)
    output_file_list = base.order_files_numerically(output_file_list)
    data_sample.bd_transition_counts = get_bd_transition_counts(model)
    return

def array_to_dict(my_array):
    """
    Convert a numpy.array object into a dictionary for convergence
    plotting.
    """
    new_dict = {}
    if my_array is None:
        return new_dict
    
    if isinstance(my_array, list):
        my_array = np.array(my_array)
    
    it = np.nditer(my_array, flags=["multi_index"])
    for x in it:
        new_dict[it.multi_index] = float(x)
    
    return new_dict

def collapse_list_of_dicts(my_list_of_dicts):
    """
    if there is a list of dictionaries, then collapse into a single 
    dictionary with 3-tuple keys.
    """
    new_dict = {}
    for i, old_dict in enumerate(my_list_of_dicts):
        for key in old_dict:
            if isinstance(key, tuple):
                new_key = (*key, i)
            elif isinstance(key, int):
                new_key = (key, i)
            else:
                raise Exception(f"Not supported key type: {key}")
            new_dict[new_key] = old_dict[key]
            
    return new_dict

def analyze_kinetics(model, analysis, max_step, k_on_state=None):
    """
    Extract relevant analysis quantities from sub-sections of the 
    data, which will later be processed for plotting.
    
    Parameters
    -----------
    model : Model()
        milestoning model object containing all milestone and 
        transition information.
        
    analysis : Analysis()
        The object which enables calculation of kinetic and 
        thermodynamic quantities.
        
    max_step_list : list
        A list of integers representing the maximum number of steps
        to analyze per anchor. Used for convergence purposes.
        
    k_on_state : int or None, default None
        If not None, then assume then this is the bound state to 
        assume for k-on calculations. A value of None will skip the
        k-on convergence..
    
    Returns
    -------
    k_on : float
        The k-on value computing using data up to the number of steps 
        in max_step_list for each milestone.
        
    k_off : float
        The k-k_off value computing using data up to the number of 
        steps in max_step_list for each milestone.
        
    N_ij : dict
        The n x n dict matrix representing how many transitions have
        occurred between milestones.
        
    R_i : dict
        An n dict representing the incubation times at each milestone.
    """
    analysis.extract_data(max_step=max_step)
    analysis.fill_out_data_samples()
    try:
        sufficient_statistics = analysis.check_extraction(silent=True)
        if sufficient_statistics:
            analysis.process_data_samples()
        else:
            analyze_bd_only(model, analysis.main_data_sample)
            
        if (k_on_state is not None) and (k_on_state in analysis.k_ons):
            k_on = analysis.k_ons[k_on_state]
        else:
            k_on = 0.0
        k_off = analysis.k_off
        main_data_sample = analysis.main_data_sample
        if isinstance(main_data_sample, mmvt_analyze.MMVT_data_sample):
            if main_data_sample.pi_alpha is None:
                pi_alpha = None
            else:
                pi_alpha = main_data_sample.pi_alpha.flatten()
                
            return k_on, k_off, main_data_sample.N_alpha_beta, \
                array_to_dict(main_data_sample.T_alpha),\
                main_data_sample.k_alpha_beta, array_to_dict(pi_alpha), \
                collapse_list_of_dicts(main_data_sample.N_i_j_alpha), \
                collapse_list_of_dicts(main_data_sample.R_i_alpha), \
                main_data_sample.N_ij, main_data_sample.R_i
        
        else:
            return k_on, k_off, {}, {}, {}, {}, {}, {}, main_data_sample.N_ij, \
                main_data_sample.R_i
        
    
    except (common_analyze.MissingStatisticsError, np.linalg.LinAlgError,
            AssertionError, ValueError) as e:
        if model.using_bd():
            analyze_bd_only(model, analysis.main_data_sample)
            
        return 0.0, 0.0, {}, {}, {}, {}, {}, {}, {}, {}

def get_mmvt_max_steps(model):
    """
    Extract the largest simulation step number for all the sims in 
    the anchors.
    """
    max_steps_all_anchors = 0
    dt = model.get_timestep()
    anchor_max_times = {}
    for anchor in model.anchors:
        max_steps = 0
        output_file_glob = os.path.join(
            model.anchor_rootdir, anchor.directory, 
            anchor.production_directory,
            anchor.md_output_glob)
        for output_filename in glob.glob(output_file_glob):
            if not os.path.exists(output_filename):
                continue
            with open(output_filename, "rb") as output_file:
                try:
                    if model.openmm_settings is not None:
                        output_file.seek(-2, os.SEEK_END)
                        while output_file.read(1) != b"\n":
                            output_file.seek(-2, os.SEEK_CUR)
                        last_line = output_file.readline().decode()
                        if last_line.startswith("#"):
                            # empty output file
                            continue
                        elif last_line.startswith("CHECKPOINT"):
                            last_line = last_line.strip().split(",")
                            if len(last_line) != 2:
                                continue
                            if re.match(r"^-?\d+\.\d{3,20}$", last_line[1]):
                                mytime = float(last_line[1])
                            else:
                                continue
                        else:
                            last_line = last_line.strip().split(",")
                            if len(last_line) != 3:
                                continue
                            if re.match(r"^-?\d+\.\d{3,20}$", last_line[2]):
                                mytime = float(last_line[2])
                            else:
                                continue
                        step = int(round(mytime / dt))
                    elif model.namd_settings is not None:
                        step = 0
                        for line in output_file:
                            line = line.decode("UTF-8")
                            if not line.startswith("SEEKR") \
                                    or len(line.strip()) == 0:
                                continue
                            elif line.startswith("SEEKR: Cell"):
                                line = line.strip().split(" ")
                                step = int(line[8].strip(","))
                                
                            elif line.startswith("SEEKR: Milestone"):
                                line = line.strip().split(" ")
                                step = int(line[10].strip(","))
                                    
                except OSError:
                    step = 0
                if step > max_steps:
                    max_steps = step
                    
                    
        anchor_max_times[anchor.index] = max_steps * dt
        max_steps = DEFAULT_NUM_POINTS * int(math.ceil(
            max_steps / DEFAULT_NUM_POINTS))
        if max_steps_all_anchors < max_steps:
            max_steps_all_anchors = max_steps
        
    conv_stride = max_steps_all_anchors // DEFAULT_NUM_POINTS
    if conv_stride == 0:
        conv_intervals = np.zeros(DEFAULT_NUM_POINTS)
    else:
        conv_intervals = np.arange(
            conv_stride, max_steps_all_anchors+conv_stride, conv_stride)
        conv_intervals = conv_intervals + DEFAULT_SKIP
    return conv_intervals, anchor_max_times

def get_elber_max_steps(model):
    """
    Extract the largest simulation step number for all the sims in 
    the anchors.
    """
    dt = model.get_timestep()
    max_steps = 0.0
    anchor_max_times = {}
    for anchor in model.anchors:
        steps = 0.0
        max_steps_this_anchor = 0.0
        output_file_glob = os.path.join(
            model.anchor_rootdir, anchor.directory, 
            anchor.production_directory,
            anchor.md_output_glob)
        for output_filename in glob.glob(output_file_glob):
            with open(output_filename, "rb") as output_file:
                try:
                    output_file.seek(-2, os.SEEK_END)
                    while output_file.read(1) != b"\n":
                        output_file.seek(-2, os.SEEK_CUR)
                    last_line = output_file.readline().decode()
                    if last_line.startswith("#"):
                        # empty output file
                        continue
                    else:
                        steps = int(last_line.strip().split(",")[1])
                except OSError:
                    steps = 0
                
        if steps > max_steps:
            max_steps = steps
        if steps > max_steps_this_anchor:
            max_steps_this_anchor = steps
            
        anchor_max_times[anchor.index] = max_steps_this_anchor * dt
        
    max_steps = DEFAULT_NUM_POINTS * int(math.ceil(
        max_steps / DEFAULT_NUM_POINTS))
    
    conv_stride = max_steps // DEFAULT_NUM_POINTS
    if conv_stride == 0:
        conv_intervals = np.zeros(DEFAULT_NUM_POINTS)
    else:
        conv_intervals = np.arange(conv_stride, max_steps+conv_stride, 
                                   conv_stride)
        conv_intervals = conv_intervals + DEFAULT_SKIP
    return conv_intervals, anchor_max_times
    
def check_milestone_convergence(model, k_on_state=None, verbose=False,
                                long_converge=True):
    """
    Calculates the key MMVT quantities N, R, and Q as a function of 
    simulation time to estimate which milestones have been 
    sufficiently sampled. 

    Quantities are pulled from the data at step intervals determined 
    by the conv_stride value with the option to skip steps from the 
    beginning of the data with the skip variable

    Parameters
    -----------
    model : Model()
        milestoning model object containing all milestone and 
        transition information.
        
    k_on_state: int or None, default None
        If not None, then assume then this is the bound state to 
        assume for k-on calculations. A value of None will skip the
        k-on convergence.
        
    verbose : bool, Default False
        Whether to provide more verbose output information.

    Returns
    -------
    k_on_conv : list
        list of calculated on rate at each convergence interval
    
    k_off_conv : list
        list of calculated off rate at each convergence interval
    
    N_ij_conv: list
        list of transition count matrix N for each convergence interval
        
    R_i_conv : list
        list of transition time matrix R for each convergence interval
        
    max_step_list : list
        list of maximum step numbers used for each convergence sample
        
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds
        
    data_sample_list : list
        A list of Data_sample objects that can be used to
        quantitatively monitor convergence.
    """
    
    data_sample_list = []
    dt = model.get_timestep()
    timestep_in_ns = (dt * unit.picosecond).value_in_unit(unit.nanoseconds)
    if model.get_type() == "mmvt":
        max_step_list, times_dict = get_mmvt_max_steps(model)
        
    elif model.get_type() == "elber":
        max_step_list, times_dict = get_elber_max_steps(model)
        
    k_off_conv = np.zeros(DEFAULT_NUM_POINTS)
    k_on_conv = np.zeros(DEFAULT_NUM_POINTS)
    # partial allows us to create a defaultdict with values that are
    # empty arrays of a certain size DEFAULT_NUM_POINTS
    N_alpha_beta_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    T_alpha_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    k_alpha_beta_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    pi_alpha_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    N_ij_alpha_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    R_i_alpha_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    N_ij_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    R_i_conv = defaultdict(functools.partial(np.zeros, DEFAULT_NUM_POINTS))
    analysis = analyze.Analysis(model, force_warning=False)
    
    if long_converge:
        interval_span = range(DEFAULT_NUM_POINTS)
    else:
        interval_span = [DEFAULT_NUM_POINTS-1]
    
    for interval_index in interval_span:
        if verbose and (interval_index % PROGRESS_UPDATE_INTERVAL == 0):
            print("Processing interval {} of {}".format(interval_index, 
                                                        DEFAULT_NUM_POINTS))
        max_step = max_step_list[interval_index]
        interval_fraction = interval_index / DEFAULT_NUM_POINTS
        k_on, k_off, N_alpha_beta, T_alpha, k_alpha_beta, pi_alpha, \
            N_ij_alpha, R_i_alpha, N_ij, R_i = analyze_kinetics(
                model, analysis, max_step, k_on_state)
        data_sample_list.append(analysis.main_data_sample)
        k_on_conv[interval_index] = k_on
        k_off_conv[interval_index] = k_off
        if interval_index == 0:
            divisor = 0.5 * max_step_list[1]
        else:
            divisor = max_step
        
        for N_alpha_beta_key in N_alpha_beta:
            N_alpha_beta_conv[N_alpha_beta_key][interval_index] \
                = N_alpha_beta[N_alpha_beta_key] #/ divisor
        for T_alpha_key in T_alpha:
            T_alpha_conv[T_alpha_key][interval_index] \
                = T_alpha[T_alpha_key] #/ divisor
        for k_alpha_beta_key in k_alpha_beta:
            k_alpha_beta_conv[k_alpha_beta_key][interval_index] \
                = k_alpha_beta[k_alpha_beta_key]
        for pi_alpha_key in pi_alpha:
            pi_alpha_conv[pi_alpha_key][interval_index] \
                = pi_alpha[pi_alpha_key]
        for N_ij_alpha_key in N_ij_alpha:
            N_ij_alpha_conv[N_ij_alpha_key][interval_index] \
                = N_ij_alpha[N_ij_alpha_key] #/ divisor
        for R_i_alpha_key in R_i_alpha:
            R_i_alpha_conv[R_i_alpha_key][interval_index] \
                = R_i_alpha[R_i_alpha_key] #/ divisor
        for N_ij_key in N_ij:
            N_ij_conv[N_ij_key][interval_index] = N_ij[N_ij_key] #/ divisor
        for R_i_key in R_i:
            R_i_conv[R_i_key][interval_index] = R_i[R_i_key] #/ divisor
    
    return k_on_conv, k_off_conv, N_alpha_beta_conv, T_alpha_conv, \
        k_alpha_beta_conv, pi_alpha_conv, N_ij_alpha_conv, R_i_alpha_conv, \
        N_ij_conv, R_i_conv, max_step_list, timestep_in_ns, data_sample_list, \
        times_dict

def plot_scalar_conv(conv_values, conv_intervals, label, title, timestep_in_ns, 
                     y_axis_logarithmic=True):
    """
    Plot convergence of off/on rate or other scalar values as a 
    function of simulation time.

    Parameters
    ----------
    conv_values : list
        list of calculated scalar values for each convergence interval
        
    conv_intervals : list
        list of convergence interval step numbers for which samples 
        are taken.
        
    label : str
        The label to give this plot.
    
    title : str
        The title of this plot.
    
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds.
        
    y_axis_logarithmic : bool, default True
        Whether the y-axis will be plotted on a logarithmic scale.

    Returns
    -------
    fig : matplotlib figure
        matplotlib figure plotting N convergence for each milestone
    ax : object
        matplotlib Axes object
    """
    #if not np.any(np.isfinite(conv_values)) or np.all(conv_values == 0):
    #    return None, None
    for i, conv_value in enumerate(conv_values):
        if not np.isfinite(conv_value) or conv_value == 0:
            conv_values[i] = np.NAN
    
    if not np.any(np.isfinite(conv_values)):
        #print("skipping key:", key, "because values aren't finite")
        return None, None
    
    fig, ax = plt.subplots()
    ax.plot(np.multiply(conv_intervals, timestep_in_ns), conv_values, 
            linestyle="-", marker="o", markersize=1)
    plt.ylabel("$"+label+"$")
    # TODO: change this to plotting over time
    plt.xlabel("time per anchor (ns)")
    plt.title(title)
    if y_axis_logarithmic:
        plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    return fig, ax

def plot_dict_conv(conv_dict, conv_intervals, label_base, unit, timestep_in_ns, 
                   skip_null=True, y_axis_logarithmic=True, title_suffix="",
                   name_base=None, draw_double=True):
    """
    Plot convergence of N_ij or R_i or other dictionary-based value 
    as a function of simulation time.

    Parameters
    ----------
    conv_dict : dict
        dict of lists of calculated off rates for each convergence 
        interval.
        
    conv_interval : list
        list of convergence interval step numbers for which samples 
        are taken.
        
    label_base : str
        The base of the label to give this plot, though the dictionary
        keys will be appended to the label.
    
    timestep_in_ns : float
        The length of the timestep in units of nanoseconds
    
    skip_null : bool, Default True
        If true, than empty convergence lists will be omitted from
        any plots.
    
    y_axis_logarithmic : bool, True
        Whether the y-axis will be plotted on a logarithmic scale.
    
    Returns
    -------
    fig_list : list
        A list of matplotlib figures plotting convergence for each 
        milestone.
        
    ax_list : list
        A list of matplotlib Axes objects.
        
    title_list : list
        A list of the plots' titles.
    
    name_list : list
        A list of the plots' file names.
    """
    
    fig_list = []
    ax_list = []
    title_list = []
    name_list = []
    if name_base is None:
        name_base = label_base
        
    for key in conv_dict:
        conv_values = conv_dict[key]
        if not np.all(np.isfinite(conv_values)):
            #print("skipping key:", key, "because values aren't finite")
            continue
        if skip_null:
            if np.linalg.norm(conv_values) < MIN_PLOT_NORM:
                #print("Skipping key:", key, "because values are too low")
                continue
        if unit == "":
            unitstr = ""
        else:
            unitstr = "(" + unit +")"
        if isinstance(key, tuple):
            # If right arrow is preferred
            #label = "$" + label_base + "_{" + "\\rightarrow".join(
            #    map(str, key)) + "}(" + unit +")$"
            #title = "$" + label_base + "_{" + "\\rightarrow".join(
            #    map(str, key)) + "}$" + title_suffix
            
            label = "$" + label_base + "_{" + ",".join(
                map(str, key)) + "}" + unitstr + "$"
            title = "$" + label_base + "_{" + ",".join(
                map(str, key)) + "}$" + title_suffix
            name = name_base + "_" + "_".join(map(str, key))
        elif isinstance(key, int):
            label = "$" + label_base + "_{" + str(key) + "}" + unitstr + "$"
            title = "$" + label_base + "_{" + str(key) + "}$" + title_suffix
            name = name_base + "_" + str(key) + ""
        else:
            raise Exception("key type not implemented: {}".format(type(key)))
        
        fig, ax1 = plt.subplots()
        x_data = np.multiply(conv_intervals, timestep_in_ns)
        ax1.plot(x_data, conv_values, 
                linestyle='-', marker="o", color="r", markersize = 1)
        ax1.set_ylabel(label, color="r")
        ax1.set_xlabel("time per anchor (ns)")
        if y_axis_logarithmic:
            plt.yscale("log", nonpositive="mask")
        ax_list.append(ax1)
        if draw_double:
            ax2 = ax1.twinx()
            ax2.set_ylabel(label + "/time(ps)", color="b")
            data_per_time = np.zeros(conv_intervals.shape)
            for i in range(len(conv_intervals)):
                data_per_time[i] = conv_values[i] / x_data[i]
            ax2.plot(x_data, data_per_time, 
                linestyle='-', marker="o", color="b", markersize = 1)
            if y_axis_logarithmic:
                plt.yscale("log", nonpositive="mask")
            ax_list.append(ax2)
        
        plt.title(title)
        fig_list.append(fig)
        title_list.append(title)
        name_list.append(name)
        plt.tight_layout()
        
    return fig_list, ax_list, title_list, name_list

def calc_transition_steps(model, data_sample):
    """
    For a given data_sample object, return the number of transitions
    and the minimum transitions between a pair of milestones.
    """
    transition_minima = []
    transition_prob_details = []
    transition_time_details = []
    for alpha, anchor in enumerate(model.anchors):
        transition_detail = {}
        transition_time_detail = {}
        if anchor.bulkstate:
            continue
        
        if model.get_type() == "mmvt":
            if data_sample.T_alpha is None:
                transition_minima.append(0)
                transition_prob_details.append(transition_detail)
                transition_time_details.append(transition_time_detail)
                continue
        
        elif model.get_type() == "elber":
            if data_sample.R_i_list[alpha] == 0.0:
                transition_minima.append(0)
                transition_prob_details.append(transition_detail)
                transition_time_details.append(transition_time_detail)
                continue
        
        if len(anchor.milestones) == 1:
            # if this is a dead-end milestone
            if model.get_type() == "mmvt":
                transition_dict = data_sample.N_alpha_beta
                k_rate_dict = data_sample.k_alpha_beta
                if len(transition_dict) == 0:
                    transition_quantity = 0
                    
                else:
                    lowest_value = 1e99
                    for key in transition_dict:
                        if key[0] == alpha and transition_dict[key] \
                                < lowest_value:
                            lowest_value  = transition_dict[key]
                    transition_quantity = lowest_value
                    
                    lowest_value = 1e99
                    for key in k_rate_dict:
                        if k_rate_dict[key] == 0.0:
                            lowest_value = 0.0
                            continue
                        if key[0] == alpha and k_rate_dict[key] \
                                < lowest_value:
                            lowest_value  = 1.0 / k_rate_dict[key]
                    transition_time = lowest_value
                    if transition_time < model.get_timestep():
                        transition_time = 0.0
                    
                    transition_detail = {(alpha,alpha):transition_quantity}
                    #transition_time_detail = {(alpha,alpha):transition_time}
                    transition_time_detail = {alpha:transition_time}
                    
            elif model.get_type() == "elber":
                raise Exception("Elber simulations cannot have one milestone.")
            
        else:
            if model.get_type() == "mmvt":
                transition_dict = data_sample.N_i_j_alpha[alpha]
                time_dict = data_sample.R_i_alpha[alpha]
                
            elif model.get_type() == "elber":
                transition_dict = data_sample.N_i_j_list[alpha]
                time_dict = {alpha: data_sample.R_i_list[alpha]}
                
            if len(transition_dict) == 0:
                transition_quantity = 0
            else:
                lowest_value = 1e99
                highest_value = 0
                for key in transition_dict:
                    if transition_dict[key] < lowest_value:
                        lowest_value  = transition_dict[key]
                    if transition_dict[key] > highest_value:
                        highest_value  = transition_dict[key]
                    transition_detail[key] = transition_dict[key]
                transition_quantity = lowest_value
                
                for key in time_dict:
                    if highest_value == 0:
                        transition_time_detail[key] = 0.0
                    else:
                        transition_time_detail[key] = time_dict[key] \
                            / highest_value
                
        transition_minima.append(transition_quantity)
        transition_prob_details.append(transition_detail)
        transition_time_details.append(transition_time_detail)
    
    return transition_minima, transition_prob_details, transition_time_details

def calc_window_rmsd(conv_values):
    """
    For a window of convergence values, compute the RMSD and average
    of those values.
    """
    if len(conv_values) == 0:
        return 0.0, 0.0
    average = np.mean(conv_values)
    RMSD_sum = 0.0
    for conv_value in conv_values:
        RMSD_sum += (conv_value - average)**2
    RMSD = np.sqrt(RMSD_sum / len(conv_values))
    return RMSD, average

def calc_RMSD_conv_amount(model, data_sample_list, window_size=None,
                               number_of_convergence_windows=1):
    """
    Calculate the RMSD convergence of window spanning a portion of
    a list of data samples.
    """
    if window_size is None:
        window_size = DEFAULT_NUM_POINTS // 2
    convergence_results = []
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        conv_list = []
        RMSD_window_conv_list = []
        for window_index in range(number_of_convergence_windows):
            backwards = number_of_convergence_windows - window_index - 1
            bound1 = len(data_sample_list) - window_size - backwards
            bound2 = len(data_sample_list) - backwards
            for data_sample in data_sample_list[bound1:bound2]:
                if model.get_type() == "mmvt":
                    if data_sample.T_alpha is None:
                        RMSD_window_conv_list.append(1e99)
                        break
                elif model.get_type() == "elber":
                    if data_sample.R_i_list[alpha] == 0.0:
                        RMSD_window_conv_list.append(1e99)
                        break
                
                if len(anchor.milestones) == 1:
                    # if this is a dead-end milestone
                    if model.get_type() == "mmvt":
                        transition_dict = data_sample.N_alpha_beta
                        T_alpha = data_sample.T_alpha[alpha]
                        lowest_value = 1e99
                        for key in transition_dict:
                            if key[0] == alpha and transition_dict[key] \
                                    < lowest_value:
                                lowest_value  = transition_dict[key]
                                if T_alpha == 0:
                                    conv_quantity = 1e99
                                else:
                                    conv_quantity = transition_dict[key] \
                                        / T_alpha
                        
                        if lowest_value == 1e99 or lowest_value == 0:
                            RMSD_window_conv_list.append(1e99)
                            break
                    else:
                        raise Exception(
                            "Elber simulations cannot have one milestone.")
                else:
                    if model.get_type() == "mmvt":
                        transition_dict = data_sample.N_i_j_alpha[alpha]
                        T_alpha = data_sample.T_alpha[alpha]
                    elif model.get_type() == "elber":
                        transition_dict = data_sample.N_i_j_list[alpha]
                        T_alpha = data_sample.R_i_list[alpha]
                        
                    lowest_value = 1e99
                    
                    for key in transition_dict:
                        if transition_dict[key] < lowest_value:
                            lowest_value  = transition_dict[key]
                            if T_alpha == 0:
                                conv_quantity = 1e99
                            else:
                                conv_quantity = transition_dict[key] / T_alpha
                    
                    if lowest_value == 1e99 or lowest_value == 0:
                        RMSD_window_conv_list.append(1e99)
                        break
                
                conv_list.append(conv_quantity)
                                
            RMSD, window_average = calc_window_rmsd(conv_list)
            if window_average == 0.0:
                RMSD_window_conv_list.append(1e99)
            else:
                fraction = RMSD / window_average
                RMSD_window_conv_list.append(fraction)
            
        max_RMSD = np.max(RMSD_window_conv_list)
        convergence_results.append(max_RMSD)
        
    return convergence_results