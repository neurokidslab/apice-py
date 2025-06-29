# %% LIBRARY

# Import necessary modules
import numpy as np 
from prettytable import PrettyTable 

# Import functions and classes from apice library
from apice.filter import * 
from apice.artifacts_structure import Artifacts 
import apice.artifacts_detection  
import apice.artifacts_structure 
from apice.io import Raw 
from apice.parameters import * 


# %% FUNCTIONS
def print_artifact_header(name):
    """
    Print a header for artifact information with a separator line.

    Args:
    - name (str): Name of the artifact.
    """
    
    # Create the header text
    header_text = f'ARTIFACTS: {name}'
    
    # Create a separator line with the same length as the header text
    separator = f'\n{"-" * len(header_text)}\n'
    
    # Print the header and separator
    print(f'{separator}{header_text}{separator}')
    
def print_data_info(raw, name, params, max_loop_num):
    """
    Display the current raw EEG data information and status.

    Args:
    - raw (object): Object containing raw EEG data.
    - name (str): Name of the rejection algorithm.
    - params (dict): Rejection algorithm settings.
    - max_loop_num (int): Maximum number of loop iterations.

    Returns:
    - None
    """

    # Mapping of algorithm names to more descriptive names
    names = {'BadElectrodes': 'BAD ELECTRODES',
            'Motion1': 'MOTION',
            'Motion2': 'MOTION',
            'Jump': 'JUMP'}

    print_artifact_header(names[name])

    # Data size information
    n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
    total_samples = n_electrodes * n_samples * n_epochs
    print(f'{n_electrodes} electrodes, {n_samples} samples, {n_epochs} epochs')
    print(f'Initial number of samples: {total_samples}')

    # Rejection settings
    if not params['keep_rejection_cause']:
        print('- The rejection cause will not be saved')
    else:
        print('- The rejection cause will be saved')

    if not params['keep_rejected_previous']:
        print('- BCT will be reset')
    else:
        print('- The previous BCT will be kept')

    # Filter settings
    if params['low_pass_freq']:
        print('- The data will be low-pass filtered before detection:', params['low_pass_freq'], 'Hz')
    if params['high_pass_freq']:
        print('- The data will be high-pass filtered before detection:', params['high_pass_freq'], 'Hz')

    # Rejection settings
    print('- The rejection algorithms will be applied')
    print('   - a maximum of', str(np.squeeze(max_loop_num)), 'times')
    print('   - or until the new rejected data is less than ', np.round(params['rejection_tolerance'], 2), '%')
    print('\n')
    return

def set_rejection_order(name):
    """
    Defines the order and parameters of rejection algorithms to be executed.

    This function determines the order and parameters of rejection algorithms to be executed based on the specified name.

    Parameters:
    - name (str): The name of the rejection algorithm.

    Returns:
    - rejection_algorithms (list): List of rejection algorithms in the order they will be executed.
    - rejection_steps (dict): Dictionary mapping step names to corresponding rejection algorithms.
    - rejection_loops (dict): Dictionary mapping step names to the number of loops for each step.
    - n_loops_max (int): Maximum number of loops to be executed.
    """
    
    # Define a dictionary to map algorithm names to their parameters
    algorithm_params = {
        'BadElectrodes': apice.parameters.BadElectrodes.PARAMS,
        'Motion1': apice.parameters.Motion1.PARAMS,
        'Motion2': apice.parameters.Motion2.PARAMS,
        'Jump': apice.parameters.Jump.PARAMS,
    }

    # Check if the specified name is in the dictionary
    if name in algorithm_params:
        detection_algorithms = algorithm_params[name]
    else:
        raise ValueError(f"Unsupported rejection algorithm name: {name}")

    # Initialize variables
    rejection_algorithms = list(detection_algorithms.keys())
    rejection_steps = {}
    rejection_loops = {}
    loop_order = []
    steps = []
    n_steps = 0

    # Extract loop order and steps from detection algorithms
    for i in np.arange(np.size(rejection_algorithms)):
        for j in np.arange(len(detection_algorithms[rejection_algorithms[i]]['loop_num'])):
            loop_order.append(detection_algorithms[rejection_algorithms[i]]['loop_num'][j])
            n_steps += 1
            steps.append(rejection_algorithms[i])

    # Set new rejection steps
    steps_new = []
    for i in np.arange(np.max(loop_order)):
        ind = np.asarray(np.where(loop_order == i + 1)).tolist()
        for j in list(ind[0]):
            steps_new.append(steps[j])

    # Sort loop algorithms
    loop_order = np.sort(loop_order)
    
    # Create dictionaries for rejection steps and loops
    for i in np.arange(len(steps_new)):
        if steps[i] in apice.artifacts_structure.Artifacts.DETECTION_ALGORITHMS:
            rejection_steps['Step_' + str(i)] = detection_algorithms[steps_new[i]]['algorithm']
            rejection_loops['Step_' + str(i)] = loop_order[i]
        else:
            rejection_steps['Step_' + str(i)] = detection_algorithms[steps_new[i]]['algorithm']
            rejection_loops['Step_' + str(i)] = loop_order[i]


    # Identify maximum loop number
    n_loops_max = np.max(loop_order)
    
    # Set rejection_algorithms
    rejection_algorithms = steps_new
    
    return rejection_algorithms, rejection_steps, rejection_loops, n_loops_max

def initialize_rejection_algorithms(raw, name, params):
    """
    Initialize parameters and variables for the rejection algorithms.

    - raw: Object containing raw EEG data and information.
    - name: Name of the rejection algorithm.
    - params: Dictionary of parameters for the rejection algorithm.

    Returns:
    - Tuple containing initialized variables and parameters for the rejection process.
    """
    
    # Set the order and the number of repetitions of the rejection algorithm
    rejection_algorithms, rejection_steps, rejection_loops, max_loop_num = set_rejection_order(name)
    n_steps_rejection = len(rejection_steps)

    # Display initial parameters and information
    print_data_info(raw, name, params, max_loop_num)

    # Create a matrix for comparing the changes in BCT after a rejection step
    BCT_rejected = {}
    for i in range(n_steps_rejection):
        BCT_rejected['Step_' + str(i + 1)] = {}

    # Initialize loop parameters    
    step_name = {}
    step_params = {}
    for i in range(n_steps_rejection):
        step_name['Step_' + str(i + 1)] = {}
        step_params['Step_' + str(i + 1)] = {}
    step_done = np.full(n_steps_rejection, False)
    
    # Set rejection step count
    rejection_step = np.zeros(n_steps_rejection)
    rejection_step_new = np.zeros(n_steps_rejection)

    return rejection_algorithms, rejection_loops, rejection_steps, rejection_step, rejection_step_new, \
        BCT_rejected, step_done, step_name, step_params, max_loop_num

def reject_artifacts(raw, rejection_algorithms, rejection_steps, rejection_loops, rejection_algorithm, rejection_step,
                    step_done, step_name, step_params, loop_num, BCT_pre, BCT_rejected, rejection_step_number,
                    rejection_step_new, loop_name):
    """
    Apply rejection algorithms to raw EEG data and update rejection information.

    - raw: Object containing raw EEG data and information.
    - rejection_algorithms: List of rejection algorithms in the order they should be executed.
    - rejection_steps: Dictionary mapping step names to rejection algorithm names.
    - rejection_loops: Dictionary mapping step names to their loop numbers.
    - rejection_algorithm: Dictionary of initialized rejection algorithm objects.
    - rejection_step: Array to store the number of rejected data points for each step.
    - step_done: Array to track whether each step has been completed.
    - step_name: Dictionary to store the name of each step.
    - step_params: Dictionary to store the parameters used for each step.
    - loop_num: Current loop number.
    - BCT_pre: Binary Channel Time matrix before rejection.
    - BCT_rejected: Dictionary to store BCT after each rejection step.
    - rejection_step_number: Current rejection step number.
    - rejection_step_new: Array to store the number of newly rejected data points for each step.
    - loop_name: Name of the current loop.

    Returns:
    - Updated rejection_algorithm, BCT, BCT_rejected, rejection_step_number, rejection_steps.
    """
    
    # Set step count
    step = 0
    
    # Loop over rejection algorithms
    for j in list(rejection_steps.keys()):
        if np.squeeze(rejection_loops[j]) == loop_num:
            step = np.sum(step_done) + 1  
            print('\n----------------------------------------------')
            print('Rejection step ', str(step), ': ', rejection_steps[j]) 

            # Get the name and class of the rejection algorithm
            name = rejection_algorithms[step - 1]
            class_ = getattr(apice.artifacts_detection, rejection_steps[j])

            # Instantiate and run the rejection algorithm
            rejection_algorithm[rejection_steps[j]] = class_(raw, config=True, name=name, loop_name=loop_name)

            # Extract BCT after rejection using the rejection algorithm
            BCT_post = rejection_algorithm[rejection_steps[j]].BCT

            # Calculate the number of rejected samples
            n_rejected_samples = rejection_algorithm[rejection_steps[j]].n_rejected_data

            # Determine if it's a post-detection algorithm and set the new rejected samples accordingly
            if class_.__name__ in apice.artifacts_structure.Artifacts.POSTDETECTION_ALGORITHMS:
                n_rejected_samples_new = n_rejected_samples
            else:
                n_rejected_samples_new = np.sum(np.logical_and(BCT_post, np.logical_not(BCT_pre)))

            # Update rejection step information
            rejection_step[rejection_step_number] = n_rejected_samples
            rejection_step_new[rejection_step_number] = n_rejected_samples_new

            # New BCT
            if class_.__name__ in apice.artifacts_structure.Artifacts.DETECTION_ALGORITHMS:
                BCT = np.logical_or(BCT_post, BCT_pre)
            else:
                BCT = BCT_post
            BCT_rejected[j] = BCT_post

            # Print rejected data
            if class_.__name__ in apice.artifacts_structure.Artifacts.DETECTION_ALGORITHMS:
                print('\nNew rejected data: ', np.round(n_rejected_samples_new / np.size(BCT) * 100, 2), '%')
            else:
                # Print rejected data
                if class_.__name__ == 'ShortBadSegments':
                    print('\nNew re-included data: ', np.round(n_rejected_samples_new / np.size(BCT) * 100, 2), '%')
                else:
                    print('\nNew rejected data: ', np.round(n_rejected_samples_new / np.size(BCT) * 100, 2), '%')

            # Get a copy of current BCT
            BCT = raw.artifacts.BCT

            # Update rejection step information
            step_done[rejection_step_number] = True

            # Get the rejection step name and parameters from the algorithm
            step_name[j] = raw.artifacts.algorithm['step_name']
            step_params[j] = raw.artifacts.algorithm['params']

            # Increment the rejection step number and update the loop number
            rejection_step_number += 1
            loop_num = rejection_loops[j]

    return rejection_algorithm, BCT, BCT_rejected, rejection_step_number, rejection_steps

def detect_artifacts(raw, rejection_algorithms, detection_steps, detection_loops, loop_num, step_done, step_name,
                    step_params, rejection_algorithm, rejection_step, BCT_pre, BCT_rejected, rejection_step_number,
                    rejection_step_new, loop_name):
    """
    Detect artifacts in raw EEG data using a series of artifact detection algorithms.

    :param raw : Object containing raw EEG data and information.
    :param rejection_algorithms: List of rejection algorithm names.
    :param detection_steps: Dictionary of artifact detection steps.
    :param detection_loops: Dictionary of loop numbers for detection steps.
    :param loop_num: Current loop number.
    :param step_done: Array indicating whether each step is done.
    :param step_name: Dictionary to store the names of each step.
    :param step_params: Dictionary to store the parameters of each step.
    :param rejection_algorithm: Dictionary to store rejection algorithms.
    :param rejection_step: Array to store the number of rejected samples at each step.
    :param BCT_pre: Binary Channel Table (BCT) before artifact detection.
    :param BCT_rejected: Dictionary to store BCT after each rejection step.
    :param rejection_step_number: Current step number.
    :param rejection_step_new: Array to store the number of newly rejected samples at each step.
    :param loop_name: Name of the current loop.

    :return: Updated rejection_algorithm, BCT_rejected, and rejection_step_number.
    """
    
    # Calculate the number of detection steps
    n_steps_detection = len(detection_steps)

    # Loop over detection steps and check if they belong to the current loop
    for j in range(n_steps_detection):
        if np.squeeze(detection_loops['Step_' + str(j)]) == loop_num:
            step = np.sum(step_done) + 1
            print(f'\n----------------------------------------------')
            print(f'Rejection step {step}: {detection_steps["Step_{step - 1}"]}')

            # Run the detection algorithm for the current detection step
            step_key = 'Step_' + str(step - 1)
            detection_algorithm_name = detection_steps[step_key]
            class_ = getattr(apice.artifacts_detection, detection_algorithm_name)
            rejection_algorithm[detection_algorithm_name] = class_(raw,
                                                                    update_summary=False,
                                                                    update_algorithm=False,
                                                                    loop_name=loop_name,
                                                                    name=rejection_algorithms[step - 1])

            # Extract BCT after detection
            BCT_post = rejection_algorithm[detection_algorithm_name].BCT

            # Get the number of rejected samples after the detection step
            n_rejected_samples = np.sum(BCT_post)
            n_rejected_samples_new = np.sum(BCT_post & ~BCT_pre)
            rejection_step[rejection_step_number] = n_rejected_samples
            rejection_step_new[rejection_step_number] = n_rejected_samples_new

            # New BCT
            BCT_rejected['Step_' + str(step - 1)] = BCT_post

            # Print rejected data
            print('\nNew rejected data: ', np.round(np.sum(BCT_post & ~BCT_pre) / np.size(BCT_post) * 100, 2), '%')

            # Generate the step key just once to use it for both step_name and step_params
            step_key = f'Step_{step - 1}'
            step_done[rejection_step_number] = True
            step_name[step_key] = raw.artifacts.algorithm['step_name']
            step_params[step_key] = raw.artifacts.algorithm['params']
            rejection_step_number += 1
    
    return rejection_algorithm, BCT_rejected, rejection_step_number

def get_loop_summary(raw, step_done, rejection_step, rejection_step_new, rejection_algorithm, name):
    """
    Generate a summary table of raw EEG data processing steps, including statistics on rejected samples.
    
    Parameters:
    - raw (raw EEG object): The raw EEG data object, expected to have an `_data` attribute and an `artifacts` attribute.
    - step_done (list/np.array): A boolean array indicating which steps have been completed.
    - rejection_step (list/np.array): An array of the cumulative number of rejected samples at each step.
    - rejection_step_new (list/np.array): An array of the number of newly rejected samples at each step.
    - rejection_algorithm (dict): A dictionary mapping step numbers to step names.
    - name (str): A descriptive name for the summary being generated.
    
    Returns:
    - PrettyTable: A formatted table summarizing the rejection steps and statistics.
    """

    # Calculate the total number of processing steps completed
    total_steps = int(np.sum(step_done))
    # Determine the total number of samples in the raw EEG data
    total_samples = np.size(raw._data)
    # Calculate the total number of rejected samples
    total_samples_rejected = np.sum(raw.artifacts.BCT)
    # Calculate the number of remaining samples after rejection
    total_samples_remaining = total_samples - total_samples_rejected
    # Calculate the percentage of rejected samples at each step
    processed_rejection_step = np.round(rejection_step / total_samples * 100, 2)
    # Calculate the percentage of newly rejected samples at each step
    processed_rejection_step_new = np.round(rejection_step_new / total_samples * 100, 2)

    # Initialize a PrettyTable for the summary of steps
    summary = PrettyTable()
    summary.field_names = ["Step No.", "Step Name", "Rejected Samples", "New Rejected Samples"]
    # Populate the table with data for each step
    for i in range(total_steps):
        step_key = f'Step_{i}'
        summary.add_row([
            f" - Step {i + 1:02d}:", 
            rejection_algorithm[step_key],
            f"{int(rejection_step[i])} ({processed_rejection_step[i]}%)",
            f"{int(rejection_step_new[i])} ({processed_rejection_step_new[i]}%)"
        ])
    # Align the columns of the table for better readability
    summary.align["Step Name"] = "l"
    summary.align.update((key, "r") for key in ["Rejected Samples", "New Rejected Samples"])

    # Print the summary along with additional rejection statistics
    rejection_percentage = total_samples_rejected / total_samples * 100
    remaining_percentage = total_samples_remaining / total_samples * 100

    print(f'\nSUMMARY: {name}')
    print(summary)
    print(f'Rejected samples: {total_samples_rejected} ({rejection_percentage:.2f}%)')
    print(f'Remaining samples: {total_samples_remaining} ({remaining_percentage:.2f}%)')

    return summary

def run_rejection_loops(raw, BCT, BCT_rejected, rejection_loops_params, loop_name):
    """
    Run loops of artifact rejection algorithms on raw EEG data until a rejection threshold is achieved
    or the maximum number of loops is reached.

    Args:
    - raw (raw EEG object): The raw EEG data object to be processed.
    - BCT (np.array): Binary Channel Time matrix representing initial artifact status.
    - BCT_rejected (dict): Dictionary to keep track of rejected samples in each loop.
    - rejection_loops_params (dict): Parameters for the rejection algorithms and loops.
    - loop_name (str): The name of the current loop for tracking purposes.

    Returns:
    - rejection_algorithm (dict): Updated dictionary of rejection algorithm states.
    - rejection_steps (dict): Updated dictionary with the status of each rejection step.
    - params (dict): Updated rejection_loops_params with additional rejection information.
    """

    # Initialize parameters and tracking variables
    params = rejection_loops_params
    threshold_achieved = False  # Condition to terminate the loops
    loop_num = 0  # Current loop count
    rejection_step_number = 0  # Step counter within each loop

    # Initialize dictionary to save rejection steps information
    rejection_algorithm = {}
    rejection_steps = {}

    # Loop until the rejection threshold is met or the maximum number of loops is reached
    while not threshold_achieved:

        BCT_pre = BCT.copy()  # Snapshot of BCT before current loop

        loop_num += 1
        print(f'\n**************** LOOP NO. {loop_num} ****************')

        # Run rejection algorithms to detect and reject artifacts
        rejection_algorithm, BCT, BCT_rejected, rejection_step_number, \
        rejection_steps = reject_artifacts(raw,
                                            params['rejection_algorithms'],
                                            params['rejection_steps'],
                                            params['rejection_loops'],
                                            rejection_algorithm,
                                            params['rejection_step'],
                                            params['step_done'],
                                            params['step_name'],
                                            params['step_params'],
                                            loop_num,
                                            BCT_pre,
                                            BCT_rejected,
                                            rejection_step_number,
                                            params['rejection_step_new'],
                                            loop_name)

        # Update the raw EEG object with the new BCT information
        raw.artifacts.BCT = BCT
        BCT = raw.artifacts.BCT.copy()  # Refresh local BCT variable with the updated raw EEG data

        # Calculate the percentage of data rejected in the current loop
        rejected_data_after_loop = np.logical_and(BCT, np.logical_not(BCT_pre))
        n_rejected_data_after_loop = np.sum(rejected_data_after_loop) / np.size(BCT) * 100
        print(f'\n>> DATA REJECTED IN LOOP {loop_num}: {n_rejected_data_after_loop:.4f}%\n')

        # Update parameters with the results from the current loop
        params.update({
            'rejected_data_after_loop': rejected_data_after_loop,
            'n_rejected_data_after_loop': n_rejected_data_after_loop,
            'rejection_step_number': rejection_step_number
        })

        # Determine if the rejection threshold or the maximum number of loops has been reached
        if (params['rejection_tolerance'] != 0 and
            n_rejected_data_after_loop <= params['rejection_tolerance']) or \
            (loop_num == params['max_loop_num']):
            threshold_achieved = True

    print('\n---End of Loops---\n')

    # Return updated information
    return rejection_algorithm, rejection_steps, params

def update_parameters_with_user_inputs(default_params, config_params):
    """
    Update the default parameters dictionary with user-input parameters.

    Args:
    - default_params: A dictionary containing default parameters.
    - config_params: A dictionary containing user-input parameters to override the defaults.
    - return: A dictionary with updated parameters.
    """
    for key in list(config_params.keys()):
        default_params[key] = config_params[key]
    return default_params


# %% REJECT BAD ELECTRODES

# The BadElectrodes class performs rejection of artifacts based on bad electrodes in raw EEG data.

class BadElectrodes:
    """
    A class for identifying and rejecting artifacts in EEG data due to bad electrodes.
    
    This class handles the processing of EEG data to identify bad electrodes that may contribute to artifacts in the signal. 
    It supports optional low-pass and high-pass filtering and keeps track of the artifact rejection process, including the cause of rejection if required.

    Attributes:
    -----------
        - params (dict): A dictionary of parameters for artifact rejection.
        - rejection_algorithm (list): List of algorithms used for rejection.
        - rejection_steps (list): List of steps involved in the rejection process.
        - summary (str): Summary of the rejection process.
    
    Args:
    -----
        - raw (EEGData): Object containing EEG data.
        - low_pass_freq (float, optional): Low pass filter cutoff frequency (Hz). Defaults to None.
        - high_pass_freq (float, optional): High pass filter cutoff frequency (Hz). Defaults to None.
        - keep_rejected_previous (bool, optional): Whether to reset BCT (Binary Channel Time matrix). Defaults to True.
        - keep_rejection_cause (bool, optional): Whether to save the cause of rejection. Defaults to False.
        - rejection_tolerance (float, optional): Tolerance level for artifact rejection. Defaults to 0.
        - limit_rejection_subject (bool, optional): Whether to limit rejection per subject. Defaults to True.
        - config (bool, optional): Configuration flag to update parameters with user inputs. Defaults to False.

    Methods:
        __init__(self, raw, low_pass_freq=None, high_pass_freq=None, keep_rejected_previous=True,
                keep_rejection_cause=False, rejection_tolerance=0, limit_rejection_subject=True, config=False)
            Initializes the BadElectrodes object with specified parameters and settings.
    """

    def __init__(self, raw, low_pass_freq=None, high_pass_freq=None, keep_rejected_previous=True,
                keep_rejection_cause=False, rejection_tolerance=0, limit_rejection_subject=True, config=False):
        """
        Initialize the BadElectrodes class for artifact rejection based on bad electrodes.
        
        Returns:
            None: This method does not return a value and is used to set up the class instance.

        """
        
        # Set object name
        name = 'BadElectrodes'

        # Initialize parameters
        self.params = {
            'low_pass_freq': low_pass_freq,              
            'high_pass_freq': high_pass_freq,        
            'keep_rejected_previous': keep_rejected_previous, 
            'keep_rejection_cause': keep_rejection_cause,    
            'rejection_tolerance': rejection_tolerance,      
            'limit_rejection_subject': limit_rejection_subject,
            'config': config  
        }


        # Get configuration (user-input parameters)
        if config:
            self.params = update_parameters_with_user_inputs(self.params, Artifacts.PARAMS)

        # Check if filtering is required based on user parameters
        do_filter = bool(self.params['low_pass_freq']) or bool(self.params['high_pass_freq'])

        # If filtering is needed, create a copy of the raw data
        if do_filter:
            raw_copy = raw.copy()

            # Apply high-pass filter if specified
            if bool(self.params['high_pass_freq']):
                Filter.remove_low_freq(raw, f_cutoff=self.params['high_pass_freq'])

            # Apply low-pass filter if specified
            if bool(self.params['low_pass_freq']):
                Filter.remove_high_freq(raw, f_cutoff=self.params['low_pass_freq'])

        # Initialize the artifacts structure if it doesn't exist
        if not hasattr(raw, 'artifacts'):
            raw.artifacts = apice.artifacts_structure.Artifacts(raw)

        # Check if previous rejection data should be kept
        if not self.params['keep_rejected_previous']:
            if hasattr(raw.artifacts, 'CCT'):
                CCT = raw.artifacts.CCT.copy()
            raw.__delattr__('artifacts')
            raw.artifacts = apice.artifacts_structure.Artifacts(raw)
            raw.artifacts.CCT = CCT

        # Copy the BCT artifacts for further processing
        BCT = raw.artifacts.BCT.copy()

        # Create a new artifacts structure for the raw data
        raw.artifacts = apice.artifacts_structure.Artifacts(raw)

        # Copy the BCT matrix for further processing
        BCT = raw.artifacts.BCT.copy()

        # Initialize variables for rejection algorithms and loops
        rejection_algorithms, rejection_loops, rejection_steps, rejection_step, rejection_step_new, \
        BCT_rejected, step_done, step_name, step_params, max_loop_num = initialize_rejection_algorithms(raw, name, self.params)

        # Store the parameters and variables in a dictionary
        rejection_loops_params = {
            'rejection_algorithms': rejection_algorithms,
            'rejection_steps': rejection_steps,
            'rejection_loops': rejection_loops,
            'rejection_step': rejection_step,
            'step_done': step_done,
            'step_name': step_name,
            'step_params': step_params,
            'rejection_step_new': rejection_step_new,
            'rejection_tolerance': rejection_tolerance,
            'max_loop_num': max_loop_num
        }

        # Loop over detection algorithms
        self.rejection_algorithm, self.rejection_steps, rejection_loops_params = \
            run_rejection_loops(raw, BCT, BCT_rejected, rejection_loops_params, name)

        # Get the original data if filtered
        if do_filter:
            raw._data = raw_copy

        # Summarize results
        step_done, rejection_step, rejection_step_new, rejection_algorithms = (
            rejection_loops_params.get(key, None) for key in 
            ['step_done', 'rejection_step', 'rejection_step_new', 'rejection_steps']
        )

        # Print summary
        self.summary = get_loop_summary(raw, step_done, rejection_step, rejection_step_new,
                                        rejection_algorithms, self.__class__.__name__)

        # Save rejected data after the rejection loop if specified
        if self.params['keep_rejection_cause']:
            raw.artifacts.BCTsr = rejection_loops_params['rejected_data_after_loop']

# %% REJECT MOTION ARTIFACTS

# The Motion class performs rejection of artifacts based on motion artifacts in raw EEG data.

class Motion:
    """
    A class dedicated to detecting and rejecting motion artifacts within EEG data.

    This class implements methods to identify and handle motion artifacts that can 
    contaminate EEG recordings. It supports configurable motion artifact rejection algorithms,
    filtering options, and the ability to retain or discard previously identified artifacts.

    Attributes:
    -----------
        - params (dict): Parameters controlling the motion artifact detection and rejection process.
        - rejection_algorithm (list): A list of rejection algorithms to be applied.
        - rejection_steps (list): The steps involved in the rejection algorithm.
        - summary (str): A summary of actions taken during the artifact rejection process.

    Args:
    -----
        - raw (EEG Data): Object containing EEG data.
        - type (int, optional): Type of motion artifact rejection method to use. Default is 1.
        - low_pass_freq (float, optional): Low-pass filter cutoff frequency in Hz. Default is None.
        - high_pass_freq (float, optional): High-pass filter cutoff frequency in Hz. Default is None.
        - keep_rejected_previous (bool, optional): Flag to keep previously rejected artifacts. Default is True.
        - keep_rejection_cause (bool, optional): Flag to save the cause of rejection. Default is False.
        - rejection_tolerance (float, optional): Tolerance threshold for rejection. Default is 0.
        - limit_rejection_subject (bool, optional): Flag to limit rejection by subject. Default is True.
        - config (bool, optional): Flag to use user-defined configuration. Default is False.
    
    Methods:
    --------
        __init__(self, raw, type=1, low_pass_freq=None, high_pass_freq=None, keep_rejected_previous=True,
                keep_rejection_cause=False, rejection_tolerance=0, limit_rejection_subject=True, config=False):
            Constructs the Motion object with specified parameters and settings.
    """
    def __init__(self, raw, type=1, low_pass_freq=None, high_pass_freq=None, keep_rejected_previous=True,
                keep_rejection_cause=False, rejection_tolerance=0, limit_rejection_subject=True, config=False):
        """
        Initialize the Motion class for EEG artifact rejection based on motion artifacts.
        
        Returns:
            - None: This method is a constructor and does not return a value.
        """
        
        # Parameters
        self.params = {
            'type': type,
            'low_pass_freq': low_pass_freq,
            'high_pass_freq': high_pass_freq,
            'keep_rejected_previous': keep_rejected_previous,
            'keep_rejection_cause': keep_rejection_cause,
            'rejection_tolerance': rejection_tolerance,
            'limit_rejection_subject': limit_rejection_subject,
            'config': config
        }

        # Get configuration (user-input parameters)
        if config:
            self.params = update_parameters_with_user_inputs(self.params, Artifacts.PARAMS)
            
        # Set object name
        name = 'Motion' + str(self.params['type'])
        
        # If the type is 2, disable keeping previously rejected data
        if self.params['type'] == 2:
            self.params['keep_rejected_previous'] = False
            
        # Check if filtering is required based on user parameters
        do_filter = bool(self.params['low_pass_freq']) or bool(self.params['high_pass_freq'])

        # If filtering is needed, create a copy of the raw data
        if do_filter:
            raw_copy = raw.copy()

            # Apply high-pass filter if specified
            if bool(self.params['high_pass_freq']):
                Filter.remove_low_freq(raw, f_cutoff=self.params['high_pass_freq'])

            # Apply low-pass filter if specified
            if bool(self.params['low_pass_freq']):
                Filter.remove_high_freq(raw, f_cutoff=self.params['low_pass_freq'])

        # Initialize the artifacts structure if it doesn't exist
        if not hasattr(raw, 'artifacts'):
            raw.artifacts = apice.artifacts_structure.Artifacts(raw)

        # Check if previous rejection data should be kept
        if not self.params['keep_rejected_previous']:
            if hasattr(raw.artifacts, 'CCT'):
                CCT = raw.artifacts.CCT.copy()
            raw.__delattr__('artifacts')
            raw.artifacts = apice.artifacts_structure.Artifacts(raw)
            raw.artifacts.CCT = CCT

        # Copy the BCT artifacts for further processing
        BCT = raw.artifacts.BCT.copy()

        # Initialize variables using tuple unpacking
        (rejection_algorithms, rejection_loops, rejection_steps, rejection_step, rejection_step_new,
        BCT_rejected, step_done, step_name, step_params, max_loop_num) = initialize_rejection_algorithms(raw, name, self.params)

        # Create a dictionary using a dictionary comprehension
        rejection_loops_params = {
            'rejection_algorithms': rejection_algorithms,
            'rejection_steps': rejection_steps,
            'rejection_loops': rejection_loops,
            'rejection_step': rejection_step,
            'step_done': step_done,
            'step_name': step_name,
            'step_params': step_params,
            'rejection_step_new': rejection_step_new,
            'rejection_tolerance': rejection_tolerance,
            'max_loop_num': max_loop_num
        }

        # Loop over detection algorithms
        self.rejection_algorithm, self.rejection_steps, rejection_loops_params = \
            run_rejection_loops(raw, BCT, BCT_rejected, rejection_loops_params, name)

        # Get the original data
        if do_filter:
            raw._data = raw_copy.copy()

        # Summarize results
        step_done, rejection_step, rejection_step_new, rejection_algorithms = (
            rejection_loops_params.get(key, None) for key in 
            ['step_done', 'rejection_step', 'rejection_step_new', 'rejection_steps']
        )

        # Print summary
        self.summary = get_loop_summary(raw, step_done, rejection_step, rejection_step_new,
                                        rejection_algorithms, self.__class__.__name__)

        # Save rejected data after the rejection loop if specified
        if self.params['keep_rejection_cause']:
            raw.artifacts.BCTsr = rejection_loops_params['rejected_data_after_loop']

# %% R3EJECT JUMP ARTIFACTS

# Class for performing artifact rejection based on jump artifacts in EEG data.

class Jump:
    """
    A class dedicated to detecting and rejecting jump artifacts in EEG data.

    Jump artifacts can be abrupt and extreme variations in the recorded EEG signal that are not 
    representative of the underlying neural activity. This class implements strategies to identify
    and manage these types of artifacts.

    Attributes:
    -----------
        - params (dict): Parameters defining the jump artifact rejection process.
        - rejection_algorithm (list): A list containing the algorithms used for artifact rejection.
        - rejection_steps (list): A list outlining the steps involved in the artifact rejection process.
        - summary (str): A summary of the steps taken and their results during the artifact rejection process.
    
    Args:
    -----
        - raw (EEGData): Object containing EEG data and information.
        - low_pass_freq (float, optional): Low-pass filter cutoff frequency in Hz. Default is None.
        - high_pass_freq (float, optional): High-pass filter cutoff frequency in Hz. Default is None.
        - keep_rejected_previous (bool, optional): Flag indicating whether to keep previously rejected data. Default is True.
        - keep_rejection_cause (bool, optional): Flag indicating whether to save the cause of rejection. Default is False.
        - rejection_tolerance (float, optional): Tolerance level for rejection. Default is 0.
        - limit_rejection_subject (bool, optional): Flag indicating whether to limit rejection by subject. Default is True.
        - config (bool, optional): Configuration flag to use user-input parameters. Default is False.

    Methods:
    --------
        __init__(self, raw, low_pass_freq=None, high_pass_freq=None, keep_rejected_previous=True,
                keep_rejection_cause=False, rejection_tolerance=0, limit_rejection_subject=True, config=False): 
            Constructs the Jump object with specified parameters for jump artifact detection and rejection.
    """

    def __init__(self, raw, low_pass_freq=None, high_pass_freq=None, keep_rejected_previous=True,
                keep_rejection_cause=False, rejection_tolerance=0, limit_rejection_subject=True, config=False):
        """
        Initializes the Jump artifact rejection process for EEG data.

        Returns:
            - None: This method is a constructor and does not return a value.
        """
        
        # Set class name
        name = 'Jump'

        # Parameters
        self.params = {
            'low_pass_freq': low_pass_freq,
            'high_pass_freq': high_pass_freq,
            'keep_rejected_previous': keep_rejected_previous,
            'keep_rejection_cause': keep_rejection_cause,
            'rejection_tolerance': rejection_tolerance,
            'limit_rejection_subject': limit_rejection_subject,
            'config': config
        }

        # Get configuration (user-input parameters)
        if config:
            self.params = update_parameters_with_user_inputs(self.params, Artifacts.PARAMS)

        # Check if filtering is required based on user parameters
        do_filter = bool(self.params['low_pass_freq']) or bool(self.params['high_pass_freq'])

        # If filtering is needed, create a copy of the raw data
        if do_filter:
            raw_copy = raw.copy()

            # Apply high-pass filter if specified
            if bool(self.params['high_pass_freq']):
                Filter.remove_low_freq(raw, f_cutoff=self.params['high_pass_freq'])

            # Apply low-pass filter if specified
            if bool(self.params['low_pass_freq']):
                Filter.remove_high_freq(raw, f_cutoff=self.params['low_pass_freq'])

        # Initialize the artifacts structure if it doesn't exist
        if not hasattr(raw, 'artifacts'):
            raw.artifacts = apice.artifacts_structure.Artifacts(raw)

        # Check if previous rejection data should be kept
        if not self.params['keep_rejected_previous']:
            if hasattr(raw.artifacts, 'CCT'):
                CCT = raw.artifacts.CCT.copy()
            raw.__delattr__('artifacts')
            raw.artifacts = apice.artifacts_structure.Artifacts(raw)
            raw.artifacts.CCT = CCT

        # Copy the BCT artifacts for further processing
        BCT = raw.artifacts.BCT.copy()

        # Initialize variables for artifact rejection
        (rejection_algorithms, rejection_loops, rejection_steps, rejection_step, rejection_step_new, 
        BCT_rejected, step_done, step_name, step_params, max_loop_num) = initialize_rejection_algorithms(raw, name, self.params)

        # Store the parameters and variables in a dictionary
        rejection_loops_params = {
            'rejection_algorithms': rejection_algorithms,
            'rejection_steps': rejection_steps,
            'rejection_loops': rejection_loops,
            'rejection_step': rejection_step,
            'step_done': step_done,
            'step_name': step_name,
            'step_params': step_params,
            'rejection_step_new': rejection_step_new,
            'rejection_tolerance': rejection_tolerance,
            'max_loop_num': max_loop_num
        }

        # Start the artifact rejection loop
        self.rejection_algorithm, self.rejection_steps, rejection_loops_params = \
            run_rejection_loops(raw, BCT, BCT_rejected, rejection_loops_params, name)

        # Restore the original EEG data if it was filtered
        if do_filter:
            raw._data = raw_copy

        # Summarize results
        step_done, rejection_step, rejection_step_new, rejection_algorithms = (
            rejection_loops_params.get(key, None) for key in 
            ['step_done', 'rejection_step', 'rejection_step_new', 'rejection_steps']
        )

        # Print summary
        self.summary = get_loop_summary(raw, step_done, rejection_step, rejection_step_new,
                                        rejection_algorithms, self.__class__.__name__)

        # Save rejected data after the rejection loop if specified
        if self.params['keep_rejection_cause']:
            raw.artifacts.BCTsr = rejection_loops_params['rejected_data_after_loop']
