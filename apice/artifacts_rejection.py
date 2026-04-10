# %% LIBRARY

# Import necessary modules
import json
import numpy as np 
import pandas as pd
from pathlib import Path
from prettytable import PrettyTable 

# Import functions and classes from apice library
from apice.filter import * 
from apice.artifacts_structure import Artifacts 
import apice.artifacts_detection  



# %% CLASS TO CREATE CONFIGURATION FILES FOR ARTIFACTS REJECTION

class ArtifactsConfiguration:

    def __init__(self):
        self.cfg = {}

    def add_algorithm_group(self, group_name, min_loops=0, max_loops=2, min_rejection=0, position=None, define_bcbt=True):

        if group_name in self.cfg:
            raise ValueError(f"The group name '{group_name}' already exists in the configuration.")

        if position is None:
            position = len(self.cfg) + 1

        self.cfg[group_name] = {}
        self.cfg[group_name]["position"] = position 
        self.cfg[group_name]["min_loops"] = min_loops
        self.cfg[group_name]["max_loops"] = max_loops 
        self.cfg[group_name]["min_rejection"] = min_rejection 
        self.cfg[group_name]["define_bcbt"] = define_bcbt
        self.cfg[group_name]['algorithms'] = {}

        # modify the position of the existing groups if the new group is added in between
        for key in self.cfg.keys():
            if key != group_name and self.cfg[key]["position"] >= position:
                self.cfg[key]["position"] += 1

    def add_algorithm(self, add_to, class_name, parameters, position=None, algorithm_name=None, post_detection=False):

        if add_to not in self.cfg.keys():
            raise ValueError(f"The group name '{add_to}' does not exist in the configuration. Please add the group first using 'add_algorithm_group' method.")
        
        # verify the class name corresponds to an existing algorithm detection class in the 'artifacts_detection' module
        if not hasattr(apice.artifacts_detection, class_name):
            raise ValueError(f"The class name '{class_name}' does not correspond to an existing algorithm detection class in the 'artifacts_detection' module.")

        # verify the parameters are valid inputs for that class
        algorithm_class = getattr(apice.artifacts_detection, class_name)    
        import inspect
        init_params = inspect.signature(algorithm_class.__init__).parameters    
        for param in parameters.keys():
            if param not in init_params:
                raise ValueError(f"The parameter '{param}' is not a valid input for the class '{class_name}'. Valid parameters are: {list(init_params.keys())}")
            
        # if algorithm name is not provided, set it to the class name plus a number corresponding to the number of existing algorithms in the group
        if algorithm_name is None:
            algorithm_name = class_name + '_' + str(len(self.cfg[add_to]['algorithms']) + 1)

        # if position is not provided, set it to the number of existing algorithms in the group plus one
        if position is None:
            position = len(self.cfg[add_to]['algorithms']) + 1
            
        # add the algorithm to the configuration
        if 'algorithms' not in self.cfg[add_to].keys():
            self.cfg[add_to]['algorithms'] = {}
        self.cfg[add_to]['algorithms'][algorithm_name] = {}
        self.cfg[add_to]['algorithms'][algorithm_name]['position'] = position
        self.cfg[add_to]['algorithms'][algorithm_name]['class_name'] = class_name
        self.cfg[add_to]['algorithms'][algorithm_name]['parameters'] = parameters
        self.cfg[add_to]['algorithms'][algorithm_name]['post_detection'] = post_detection

    def save_to_json(self, file_path):
        import json
        with open(file_path, 'w') as f:
            json.dump(self.cfg, f, indent=4)


    def check_configuration(self):
        # This function checks the validity of the configuration
        for group_name, group_info in self.cfg.items():
            if 'position' not in group_info.keys():
                raise ValueError(f"The group '{group_name}' does not have a 'position' key. Please add a position for the group in the configuration.")
            if 'min_loops' not in group_info.keys():
                raise ValueError(f"The group '{group_name}' does not have a 'min_loops' key. Please add a minimum number of loops for the group in the configuration.")
            if 'max_loops' not in group_info.keys():
                raise ValueError(f"The group '{group_name}' does not have a 'max_loops' key. Please add a maximum number of loops for the group in the configuration.")
            if 'min_rejection' not in group_info.keys():
                raise ValueError(f"The group '{group_name}' does not have a 'min_rejection' key. Please add a minimum rejection threshold for the group in the configuration.")
            if 'define_bcbt' not in group_info.keys():
                raise ValueError(f"The group '{group_name}' does not have a 'define_bcbt' key. Please add a boolean to specify whether to define BT and BC based on BCT for the group in the configuration.")
            if 'algorithms' not in group_info.keys():
                raise ValueError(f"The group '{group_name}' does not have an 'algorithms' key. Please add an algorithms dictionary for the group in the configuration.")
            for algorithm_name, algorithm_info in group_info['algorithms'].items():
                if 'position' not in algorithm_info.keys():
                    raise ValueError(f"The algorithm '{algorithm_name}' in group '{group_name}' does not have a 'position' key. Please add a position for the algorithm in the configuration.")
                if 'class_name' not in algorithm_info.keys():
                    raise ValueError(f"The algorithm '{algorithm_name}' in group '{group_name}' does not have a 'class_name' key. Please add a class name for the algorithm in the configuration.")
                if 'parameters' not in algorithm_info.keys():
                    raise ValueError(f"The algorithm '{algorithm_name}' in group '{group_name}' does not have a 'parameters' key. Please add a parameters dictionary for the algorithm in the configuration.")
                if 'post_detection' not in algorithm_info.keys():
                    raise ValueError(f"The algorithm '{algorithm_name}' in group '{group_name}' does not have a 'post_detection' key. Please add a post_detection boolean for the algorithm in the configuration.") 
                # check if the class name corresponds to an existing algorithm detection class in the 'artifacts_detection' module
                if not hasattr(apice.artifacts_detection, algorithm_info['class_name']):
                    raise ValueError(f"The class name '{algorithm_info['class_name']}' for algorithm '{algorithm_name}' in group '{group_name}' does not correspond to an existing algorithm detection class in the 'artifacts_detection' module.") 
                # check if the parameters are valid inputs for that class
                algorithm_class = getattr(apice.artifacts_detection, algorithm_info['class_name'])  
                import inspect
                init_params = inspect.signature(algorithm_class.__init__).parameters
                for param in algorithm_info['parameters'].keys():
                    if param not in init_params:
                        raise ValueError(f"The parameter '{param}' for algorithm '{algorithm_name}' in group '{group_name}' is not a valid input for the class '{algorithm_info['class_name']}'. Valid parameters are: {list(init_params.keys())}") 
                    

    def load_from_json(self, file_path):
        import json
        with open(file_path, 'r') as f:
            self.cfg = json.load(f)
        return self

    def set_configuration(self, cfg):
        self.cfg = cfg


def concatenate_configurations(list_of_cfgs):
    # This function can be implemented to concatenate two configurations, for example to add a new group of algorithms to an existing configuration
    artcfg = ArtifactsConfiguration()
    for cfg in list_of_cfgs:
        for group_name, group_info in cfg.items():
            if group_name not in artcfg.cfg.keys():
                artcfg.add_algorithm_group(group_name, min_loops=group_info['min_loops'], max_loops=group_info['max_loops'], min_rejection=group_info['min_rejection'], position=None)
            for algorithm_name, algorithm_info in group_info['algorithms'].items():
                artcfg.add_algorithm(group_name, algorithm_info['class_name'], algorithm_info['parameters'], position=algorithm_info['position'], algorithm_name=algorithm_name, post_detection=algorithm_info['post_detection'])
    return artcfg


# %% CLASS TO CREATE A SUMMARY TABLE FOR THE ARTIFACT REJECTION STEPS

class SummaryTable:

    def __init__(self):
        self.table = pd.DataFrame(columns=["Group", "Algorithm", "Loop", "Post Detection", "Step Rejection (%)", "New Rejection (%)", "Total Rejection (%)"])

    def add_row(self, group, algorithm, loop, post_detection, step_rejection, new_rejection, total_rejection):
        new_row = pd.DataFrame({
            "Group": [group],
            "Algorithm": [algorithm],
            "Loop": [loop],
            "Post Detection": [post_detection],
            "Step Rejection (%)": [step_rejection],
            "New Rejection (%)": [new_rejection],
            "Total Rejection (%)": [total_rejection]
        })
        self.table = pd.concat([self.table, new_row], ignore_index=True)

    def __str__(self):
        # convert to a pretty table for better visualization and return the string representation of the table
        print_table = PrettyTable()
        print_table.field_names = self.table.columns.tolist()
        for _, row in self.table.iterrows():
            print_table.add_row(row.tolist())
        return print_table.get_string()
    
    def remove_rows(self, group, loop, post_detection=None):
        if post_detection is None:
            self.table = self.table[~((self.table['Group'] == group) & (self.table['Loop'] == loop))]
        else:
            self.table = self.table[~((self.table['Group'] == group) & (self.table['Loop'] == loop) & (self.table['Post Detection'] == post_detection))]
    

# %% FUNCTION TO RUN ARTIFACT REJECTION ALGORITHMS

def run_algorithms(raw, cfg: str | Path | dict = None, force_cfg=False, l_freq=None, h_freq=None):
    """
    Run artifact rejection algorithms on raw EEG data based on a provided configuration.
    Args:    
    - raw: Object containing raw EEG data and information. 
    - cfg: Configuration dictionary specifying the groups and algorithms to run, or path to a JSON file containing the configuration. If None, the default configuration will be used.
    """
    # If the raw object does not contain an artifacts attribute raise en error
    if not (hasattr(raw, 'artifacts') and isinstance(raw.artifacts, Artifacts)):
        raise ValueError("The raw object must contain an 'artifacts' attribute of type 'Artifacts'. Please set up the artifacts structure for the raw object before running the artifact rejection algorithms.")    
    
    # Set the configuration based on the input type (None, JSON file path, or dictionary)
    if cfg is None:
        # get the path of the default configuration file
        default_cfg_path = Path(__file__).parent / "default_cfg" / 'artifacts_config.json'
        artcfg = ArtifactsConfiguration().load_from_json(default_cfg_path)
    elif isinstance(cfg, (str, Path)):
        artcfg = ArtifactsConfiguration().load_from_json(cfg)
    elif isinstance(cfg, dict):
        artcfg = ArtifactsConfiguration()
        artcfg.set_configuration(cfg)
    else:
        raise ValueError("The 'cfg' parameter must be either a dictionary, a string, or a Path object. If it is a string or Path, it should point to a JSON file containing the configuration.")

    # check the validity of the configuration
    artcfg.check_configuration()
    cfg = artcfg.cfg

    # Sort the groups based on their position in the configuration
    sorted_groups = sorted(cfg.items(), key=lambda x: x[1]['position'])

    # Initialize a summary table for the artifact rejection steps
    summary_table = SummaryTable()

    # If filter frequencies are provided, apply the filters to the raw data before running the algorithms
    if l_freq is not None or h_freq is not None:
        data_org = raw._data.copy()
        Filter(raw, l_freq=l_freq, h_freq=h_freq)
        
    # Loop through each group and run the specified algorithms
    for group_name, group_info in sorted_groups:
        
        print("\n" + "="*50)
        print(f"Artifact rejection - Running group: {group_name}")
        print("="*50)
        algorithms = group_info.get('algorithms', {})
        
        # separete the algorithms in detection and post-detection algorithms and sort them based on their position in the configuration
        sorted_algorithms = sorted(algorithms.items(), key=lambda x: x[1]['position'])
        algorithms_detection = [alg for alg in sorted_algorithms if not alg[1]['post_detection']]
        algorithms_post_detection = [alg for alg in sorted_algorithms if alg[1]['post_detection']]

        print(f"\nComputing the detection algorithms for the group {group_name}...")
        print("-"*30)
        alg_obj = []
        for algorithm_name, algorithm_info in algorithms_detection:
            class_name = algorithm_info['class_name']
            parameters = algorithm_info['parameters']
            parameters["name"] = algorithm_name
            parameters["group_name"] = group_name
            print(f"\n > Group {group_name} - Algorithm: {algorithm_name} (Class: {class_name})")
            
            val_update = parameters.get("update_artifacts", False)
            if val_update and not force_cfg:
                raise ValueError(f"The algorithm '{algorithm_name}' in group '{group_name}' is set to update the artifacts directly. This is not allowed as it can lead to unintended consequences. Please set 'update_artifacts' to False in the configuration for this algorithm or set 'force_cfg' to True to override this check.")
            if val_update and force_cfg:
                print(f"Warning: The algorithm '{algorithm_name}' in group '{group_name}' is set to update the artifacts directly. This can lead to unintended consequences. Please make sure you understand the implications of this setting.")
            parameters["update_artifacts"] = val_update
            
            class_ = getattr(apice.artifacts_detection, class_name)
            alg_obj_i = class_(**parameters)
            
            if hasattr(alg_obj_i, 'compute'):
                alg_obj_i.compute(raw)
            alg_obj.append(alg_obj_i)

        print("\nRejecting data based on the computed algorithms...")
        print("-"*30)
        max_loops = group_info['max_loops']
        min_loops = group_info['min_loops']
        min_rejection = group_info['min_rejection']
        new_rej = np.inf

        for loop_counter in range(1, max_loops + 1):
            bct_new = np.zeros(raw.artifacts.BCT.shape, dtype=bool)

            for (algorithm_name, algorithm_info), alg_obj_i in zip(algorithms_detection, alg_obj):
                
                # run the algorithm and get the new BCT
                print(f"\n > Group {group_name} - Loop {loop_counter} - Algorithm: {algorithm_name} (Class: {class_name})")
                raw, bct = alg_obj_i.reject(raw)
                perc_detected = np.sum(bct) / np.size(bct) * 100
                perc_new_detected = np.sum(np.logical_and(bct, np.logical_not(raw.artifacts.BCT))) / np.size(bct) * 100
                perc_total_detected = np.sum(np.logical_or(bct, raw.artifacts.BCT)) / np.size(bct) * 100
                bct_new = np.logical_or(bct_new, bct)
                
                # add a row to the summary table for this algorithm and loop
                summary_table.add_row(group_name, algorithm_name, loop_counter, False, f"{perc_detected:.2f}%", f"{perc_new_detected:.2f}%", f"{perc_total_detected:.2f}%")
            
            new_rej = np.sum(np.logical_and(bct_new, np.logical_not(raw.artifacts.BCT)))/np.size(bct_new)*100
            print(f" > Total newly rejected data after loop {loop_counter}: {new_rej:.2f}%")

            # if the new rejection is below the minimum rejection threshold and we have reached the minimum number of loops, we can stop the loop
            if new_rej < min_rejection and loop_counter > min_loops:
                print(f"Stopping the loop as the new rejection is below the minimum threshold of {min_rejection}% and we have reached the minimum number of loops of {min_loops}.")
                print(f"Rejection for group '{group_name}' and loop {loop_counter} will not be considered neither added to the summary table.")
                # remove the rows in the summary table corresponsig to the algorithms applied in this loop
                summary_table.remove_rows(group_name, loop_counter, post_detection=False)
                break

            # Update the BCT with the new detected artifacts
            raw.artifacts.BCT = np.logical_or(raw.artifacts.BCT, bct_new)
        
        print("\nRunning post-detection algorithms...")
        print("-"*30)
        for algorithm_name, algorithm_info in algorithms_post_detection:
            class_name = algorithm_info['class_name']
            parameters = algorithm_info['parameters']
            parameters["name"] = algorithm_name
            parameters["group_name"] = group_name
            print(f"\n > Group {group_name} - Algorithm: {algorithm_name} (Class: {class_name})")

            val_update = parameters.get("update_artifacts", True)
            if not val_update and not force_cfg:
                raise ValueError(f"The post-detection algorithm '{algorithm_name}' in group '{group_name}' is set to not update the artifacts directly. This is not allowed as it can lead to unintended consequences. Please set 'update_artifacts' to True in the configuration for this algorithm or set 'force_cfg' to True to override this check.")
            if not val_update and force_cfg:
                print(f"Warning: The post-detection algorithm '{algorithm_name}' in group '{group_name}' is set to not update the artifacts directly. This is being overridden by 'force_cfg'.")
            parameters["update_artifacts"] = val_update

            class_ = getattr(apice.artifacts_detection, class_name)
            alg_obj_i = class_(**parameters)
            raw, bct = alg_obj_i.reject(raw)
            perc_post_detected = np.sum(bct) / np.size(bct) * 100

            # # Update the BCT with the new detected artifacts
            # raw.artifacts.BCT = np.logical_or(raw.artifacts.BCT, bct)

            # add a row to the summary table for this algorithm and loop
            summary_table.add_row(group_name, algorithm_name, None, True, None, None, f"{perc_post_detected:.2f}%")


        # define BT and BC based on BCT
        if group_info['define_bcbt']:
            print("\nDefining BT and BC based on BCT...")
            print("-"*30)
            raw.artifacts.define_bcbt()
    
    if l_freq is not None or h_freq is not None:
        raw._data = data_org
                    
    print("\nSummary of artifact rejection:")
    print("-"*30)
    print(summary_table)