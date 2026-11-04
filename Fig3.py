# Combined Script for Multi-Panel Figure (v2: Layout & Viz Fixes)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Import GridSpec
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter # For tick formatting
import numpy as np
import os
import joblib # Make sure joblib is imported
import warnings # To suppress warnings if needed
import re # For extracting numbers from checkpoint strings
import glob # For finding checkpoint files
import collections # For defaultdict definition
from sklearn.decomposition import PCA
import pandas as pd # Import Pandas
from cycler import cycler


# --- Define missing factory for joblib loading (Needed if data saved with older joblib/defaultdict) ---
# If your joblib files load fine without this, you might comment it out.
nested_dict_factory = collections.defaultdict


# --- Matplotlib Styling (Consistent style) ---
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['xtick.minor.width'] = 0.8
mpl.rcParams['ytick.minor.width'] = 0.8
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10 # Adjusted default size for potentially smaller subplots
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['grid.color'] = '#cccccc'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.6



# Define colors/styles (Consolidated)
# Using a colorblind-friendly palette with high contrast
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
markers = ['o', 's', '^', 'D', 'v', '<']
linestyles = ['-', '--', '-.', ':']

# Style Mapping: [T_TomQA, L_TomQA, T_Class, L_Class]
run_plot_styles = {
    "Transformer (QSlice)":     {'color': colors[0], 'marker': markers[0], 'linestyle': linestyles[0], 'lw': 2.0},
    "LSTM (QSlice)":            {'color': colors[1], 'marker': markers[1], 'linestyle': linestyles[0], 'lw': 2.0},
    "Transformer (Classical)": {'color': '#A0CFE8', 'marker': markers[2], 'linestyle': linestyles[1], 'lw': 1.0},
    "LSTM (Classical)":        {'color': '#FABD6F', 'marker': markers[3], 'linestyle': linestyles[1], 'lw': 1.0},
}


# --- Helper Functions (Consolidated & Deduped) ---

def _project_to_simplex(data):
    """Projects data points onto a 2D simplex."""
    if data.shape[1] < 3: return data[:, 0], data[:, 1] if data.shape[1] > 1 else data[:, 0]
    v1=np.array([0, 0]); v2=np.array([1, 0]); v3=np.array([0.5, np.sqrt(3)/2])
    x_proj = data[:, 1] + data[:, 2] * 0.5; y_proj = data[:, 2] * (np.sqrt(3) / 2)
    return x_proj, y_proj

def transform_for_alpha(weights, min_alpha=0.1, transformation='log'):
    """Transforms weights to alpha values for plotting."""
    epsilon = 1e-10; weights = np.asarray(weights)
    if transformation == 'log':
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning); transformed = np.log10(weights + epsilon) - np.log10(epsilon)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    elif transformation == 'sqrt':
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning); transformed = np.sqrt(weights)
        transformed = np.nan_to_num(transformed, nan=0.0)
    elif transformation == 'cbrt': transformed = np.cbrt(weights)
    else: transformed = weights # Linear
    # Clip and normalize alpha values
    if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)): transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    max_val, min_val = np.max(transformed), np.min(transformed)
    if max_val > min_val: normalized = (transformed - min_val) / (max_val - min_val)
    else: normalized = np.ones_like(transformed) * 0.5 # Avoid division by zero if all values are same
    normalized = np.clip(np.nan_to_num(normalized, nan=0.5), 0, 1) # Final clip and NaN handling
    return min_alpha + (1.0 - min_alpha) * normalized

def load_ground_truth(run_dir: str, filename: str = 'ground_truth_data.joblib') -> dict | None:
    """Loads the ground truth data for a specific run."""
    ground_truth_filepath = os.path.join(run_dir, filename)
    if not os.path.exists(ground_truth_filepath): print(f"Error: GT file not found: {ground_truth_filepath}"); return None
    try:
        ground_truth_data = joblib.load(ground_truth_filepath)
        # print(f"Loaded GT data: {ground_truth_filepath}") # Less verbose
        if not isinstance(ground_truth_data, dict):
            print(f"Warning: Loaded GT data is not a dictionary ({type(ground_truth_data)}).")
            return None
        if not all(k in ground_truth_data for k in ['beliefs', 'probs']): print(f"Warning: GT data missing keys 'beliefs' or 'probs'.")
        # Ensure beliefs and probs are numpy arrays
        if 'beliefs' in ground_truth_data:
            try: ground_truth_data['beliefs'] = np.asarray(ground_truth_data['beliefs'])
            except Exception as e: print(f"Warning: Could not convert GT beliefs to array: {e}"); ground_truth_data['beliefs'] = None
        if 'probs' in ground_truth_data:
            try: ground_truth_data['probs'] = np.asarray(ground_truth_data['probs'])
            except Exception as e: print(f"Warning: Could not convert GT probs to array: {e}"); ground_truth_data['probs'] = None
        # Check if conversion failed or keys were missing
        if ground_truth_data.get('beliefs') is None or ground_truth_data.get('probs') is None:
             print("Error: GT beliefs or probs are missing or failed conversion.")
             return None
        return ground_truth_data
    except AttributeError as e:
         # Simplified error handling for missing factory
         if 'nested_dict_factory' in str(e) or 'defaultdict' in str(e): print(f"Error loading {ground_truth_filepath}: Missing 'defaultdict' definition during load. Ensure it's defined if needed.")
         else: print(f"AttributeError loading {ground_truth_filepath}: {e}")
         return None
    except Exception as e: print(f"Error loading GT file {ground_truth_filepath}: {e}"); return None

def load_specific_checkpoint_data(run_dir: str, is_markov3_run: bool, target_ckpt_str: str) -> dict | None:
    """
    Loads the data dictionary for a single, specific checkpoint file.
    Performs basic validation and ensures relevant arrays are numpy arrays.
    Used primarily for loading data for the visualization panels AFTER checkpoints are selected.
    """
    if target_ckpt_str is None: print("Error: target_ckpt_str cannot be None."); return None
    filename = f"markov3_checkpoint_{target_ckpt_str}.joblib" if is_markov3_run else f"checkpoint_{target_ckpt_str}.joblib"
    predictions_filepath = os.path.join(run_dir, filename)
    if not os.path.exists(predictions_filepath):
        # print(f"Debug: Checkpoint file not found: {predictions_filepath}"); # Debug print
        return None
    try:
        single_ckpt_data = joblib.load(predictions_filepath)
        # Basic validation and conversion
        if isinstance(single_ckpt_data, dict):
            for layer, data in single_ckpt_data.items():
                if isinstance(data, dict):
                    # Ensure arrays are numpy arrays
                    for key, value in data.items():
                        if isinstance(value, list):
                            try: data[key] = np.array(value)
                            except Exception: pass # Ignore if conversion fails
                    # Ensure 'predicted_beliefs' is numpy array if present
                    if 'predicted_beliefs' in data and not isinstance(data['predicted_beliefs'], np.ndarray):
                        try: data['predicted_beliefs'] = np.asarray(data['predicted_beliefs'])
                        except Exception: data['predicted_beliefs'] = None # Set to None if conversion fails
        return single_ckpt_data
    except AttributeError as e:
         if 'nested_dict_factory' in str(e) or 'defaultdict' in str(e): print(f"Error loading {predictions_filepath}: Missing 'defaultdict' definition during load. Ensure it's defined if needed.")
         else: print(f"AttributeError loading {predictions_filepath}: {e}")
         return None
    except Exception as e: print(f"Error loading checkpoint file {predictions_filepath}: {e}"); return None

def _get_plotting_params(experiment_name: str) -> dict:
    """Returns plotting parameters based on experiment type."""
    params = {
        'point_size': {'truth': 0.15, 'pred': 0.05},
        'min_alpha': 0.1,
        'transformation': 'cbrt', # 'log', 'sqrt', 'cbrt', or 'linear'
        'use_pca': False,
        'project_to_simplex': False, # Only works if not using PCA and dims >= 3
        'inds_to_plot': [1, 2], # *** REVERTED to [1, 2] as per original script ***
        'com': False # Placeholder if needed later
    }
    # Example: Add specific settings for Markov3 if needed
    # if "Markov3" in experiment_name:
    #     params['project_to_simplex'] = True # e.g., Maybe simplex is default for Markov3
    # print(f"Using plotting params: {params}") # Less verbose
    return params

def _calculate_plot_coords( beliefs_to_plot: np.ndarray, gt_beliefs_for_pca: np.ndarray, use_pca: bool, project_to_simplex: bool, inds_to_plot: list, pca_instance: PCA = None ):
    """Calculates plot coordinates based on strategy. (Using robust version)"""
    x_plot, y_plot = None, None; current_pca = pca_instance
    if beliefs_to_plot is None or not isinstance(beliefs_to_plot, np.ndarray) or beliefs_to_plot.size == 0: return None, None, current_pca

    n_samples, n_dims = beliefs_to_plot.shape
    effective_inds_to_plot = list(inds_to_plot) # Make a copy

    # Ensure indices are valid for the number of dimensions
    if n_dims <= 0: return None, None, current_pca # Cannot plot if no dimensions
    if max(effective_inds_to_plot) >= n_dims:
        original_inds = list(effective_inds_to_plot)
        effective_inds_to_plot = [0, 1] if n_dims >= 2 else [0, 0]
        print(f"Warning: inds_to_plot ({original_inds}) out of bounds for dim {n_dims}. Using {effective_inds_to_plot}.")

    if use_pca:
        if current_pca is None: # Fit PCA if not provided
            n_components = max(2, max(effective_inds_to_plot) + 1) # Need at least 2 components
            n_components = min(n_components, n_dims, n_samples) # Cannot exceed dims or samples
            if n_components >= 2:
                current_pca = PCA(n_components=n_components)
                try:
                    # Ensure GT beliefs used for fitting have enough samples
                    if gt_beliefs_for_pca is not None and gt_beliefs_for_pca.shape[0] >= n_components:
                        current_pca.fit(gt_beliefs_for_pca)
                        beliefs_proj = current_pca.transform(beliefs_to_plot)
                        # Adjust indices if needed after projection
                        if max(effective_inds_to_plot) >= n_components: effective_inds_to_plot = [0, 1]
                        x_plot = beliefs_proj[:, effective_inds_to_plot[0]]
                        y_plot = beliefs_proj[:, effective_inds_to_plot[1]]
                    else:
                         print(f"Warning: Not enough samples ({gt_beliefs_for_pca.shape[0] if gt_beliefs_for_pca is not None else 'None'}) in GT data to fit PCA with {n_components} components. Skipping PCA.")
                         use_pca = False; current_pca = None # Fallback
                except Exception as e: print(f"Error during PCA fitting/transform: {e}. Plotting raw dims."); use_pca = False; current_pca = None
            else: use_pca = False # Not enough dimensions/samples for PCA
        else: # Transform using existing PCA
            try:
                n_components = current_pca.n_components_
                if max(effective_inds_to_plot) >= n_components: effective_inds_to_plot = [0, 1] # Adjust if needed
                beliefs_proj = current_pca.transform(beliefs_to_plot)
                x_plot = beliefs_proj[:, effective_inds_to_plot[0]]
                y_plot = beliefs_proj[:, effective_inds_to_plot[1]]
            except Exception as e: print(f"Error transforming with existing PCA: {e}. Plotting raw dims."); use_pca = False # Fallback

    # If PCA wasn't used or failed, try simplex projection or direct indexing
    if x_plot is None:
        if project_to_simplex and n_dims >= 3:
            x_plot, y_plot = _project_to_simplex(beliefs_to_plot)
        else: # Default to direct indexing
            if n_dims == 1: effective_inds_to_plot = [0, 0] # Plot dim 0 vs itself if only 1D
            x_plot = beliefs_to_plot[:, effective_inds_to_plot[0]]
            # Ensure y index exists
            y_plot_idx = effective_inds_to_plot[1] if len(effective_inds_to_plot) > 1 and effective_inds_to_plot[1] < n_dims else effective_inds_to_plot[0]
            y_plot = beliefs_to_plot[:, y_plot_idx]

    return x_plot, y_plot, current_pca

def _plot_beliefs_on_ax( ax: plt.Axes, x_plot: np.ndarray, y_plot: np.ndarray, colors_rgba: np.ndarray, point_size: float ):
    """Plots points on axes, turns axis off."""
    plotted_something = False
    if x_plot is not None and y_plot is not None and colors_rgba is not None and x_plot.size > 0 and y_plot.size > 0 and colors_rgba.size > 0:
        # Ensure color array matches points after potential filtering/errors
        if colors_rgba.shape[0] == x_plot.shape[0]:
            ax.scatter(x_plot, y_plot, color=colors_rgba, s=point_size, rasterized=True, marker='.')
            plotted_something = True
        else: print(f"Warning: Mismatch points ({x_plot.shape[0]}) vs colors ({colors_rgba.shape[0]}) in _plot_beliefs_on_ax. Skipping scatter.")
    # Don't add text if simply no data to plot (x_plot/y_plot are None)
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
    ax.set_axis_off()
    # Add error text only if coordinates existed but plotting failed (e.g., color mismatch)
    if not plotted_something and (x_plot is not None or y_plot is not None):
        ax.text(0.5, 0.5, "Plot Error", ha='center', va='center', transform=ax.transAxes, fontsize=7, color='orange')
    # Add N/A text if coordinates were None from the start
    elif x_plot is None and y_plot is None:
        ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes, fontsize=7, color='red')


def extract_metric_vs_ckpt_data(run_dir: str, is_markov3_run: bool, target_layer: str, metric_key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extracts checkpoint indices and a specific metric for a target layer.
    Uses direct joblib.load for robustness in finding the metric, similar to standalone script.
    """
    checkpoints = []
    metric_values = []
    ckpt_pattern = f"markov3_checkpoint_*.joblib" if is_markov3_run else f"checkpoint_*.joblib"
    # Use sorted glob to process checkpoints in order
    ckpt_files = sorted(glob.glob(os.path.join(run_dir, ckpt_pattern)))
    if not ckpt_files: print(f"Warning: No files found matching: {os.path.join(run_dir, ckpt_pattern)}"); return None, None

    print(f"  (extract_metric) Found {len(ckpt_files)} files matching pattern in {run_dir}") # Debug print
    for f in ckpt_files:
        match = re.search(r'_(\d+)\.joblib$', os.path.basename(f))
        if not match: continue
        try: ckpt_idx = int(match.group(1))
        except ValueError: continue

        try:
            # *** Use direct joblib.load like in the standalone script ***
            ckpt_data = joblib.load(f)

            # Check if loaded data is dict and contains the target layer and metric
            if isinstance(ckpt_data, dict) and target_layer in ckpt_data:
                layer_data = ckpt_data[target_layer]
                if isinstance(layer_data, dict) and metric_key in layer_data:
                    metric_value = layer_data[metric_key]
                    processed_value = None
                    # --- Processing Logic (same as before) ---
                    if isinstance(metric_value, np.ndarray):
                        if metric_value.size > 0:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                if metric_key == 'rmse':
                                    processed_value = float(np.sqrt(np.nanmean(metric_value**2)))
                                elif metric_key in ['dist', 'r2', 'val_loss_mean', 'val_loss']:
                                     processed_value = float(np.nanmean(metric_value))
                                else: # Default to mean if unknown array metric
                                     processed_value = float(np.nanmean(metric_value))
                        # else: skip empty arrays
                    elif isinstance(metric_value, (int, float, np.number)): # Handle scalars
                        processed_value = float(metric_value)
                    # else: skip other types

                    if processed_value is not None and np.isfinite(processed_value):
                        checkpoints.append(ckpt_idx); metric_values.append(processed_value)
                    # else: print(f"Debug: Invalid processed value for {metric_key} in {f}") # Debug
                # else: print(f"Debug: Metric '{metric_key}' or layer data dict not found in {f}") # Debug
            # else: print(f"Debug: Target layer '{target_layer}' not found or ckpt_data not dict in {f}") # Debug

        # Catch errors during loading or processing for this specific file
        except FileNotFoundError:
             print(f"Error: File not found during direct load: {f}")
        except AttributeError as e:
             if 'nested_dict_factory' in str(e) or 'defaultdict' in str(e): print(f"Error loading {f}: Missing 'defaultdict' definition during load.")
             else: print(f"AttributeError loading/processing file {f}: {e}")
        except Exception as e:
            print(f"Error loading/processing file {f} for metric '{metric_key}': {e}")

    if not checkpoints: print(f"Warning: No valid data extracted for metric '{metric_key}' in layer '{target_layer}' for {run_dir} after checking {len(ckpt_files)} files."); return None, None

    print(f"  (extract_metric) Extracted {len(checkpoints)} valid data points for {metric_key}.") # Debug print
    checkpoints = np.array(checkpoints); metric_values = np.array(metric_values)
    # Data should already be sorted due to sorted glob, but explicit sort is safer
    sort_indices = np.argsort(checkpoints)
    return checkpoints[sort_indices], metric_values[sort_indices]


def find_checkpoint_str(run_dir: str, is_markov3_run: bool, first: bool = False) -> str | None:
    """Finds the string representation of the highest or lowest checkpoint index available."""
    ckpt_pattern = f"markov3_checkpoint_*.joblib" if is_markov3_run else f"checkpoint_*.joblib"
    ckpt_files = glob.glob(os.path.join(run_dir, ckpt_pattern))
    if not ckpt_files: return None
    available_indices = []
    for f in ckpt_files:
        match = re.search(r'_(\d+)\.joblib$', os.path.basename(f))
        if match:
            try: available_indices.append(int(match.group(1)))
            except ValueError: pass
    if not available_indices: return None
    target_idx = min(available_indices) if first else max(available_indices)
    return str(target_idx)


def extract_layer_metrics(checkpoint_data: dict, target_metrics: list[str]) -> dict | None:
    """ Extracts specified metrics for all layers from loaded checkpoint data, processing values. """
    if checkpoint_data is None: return None
    layer_metrics = collections.defaultdict(dict)
    if isinstance(checkpoint_data, dict):
        for layer_name, layer_data in checkpoint_data.items():
            # Skip if layer_data is not a dictionary (might be unexpected format)
            if not isinstance(layer_data, dict): continue

            for metric_key in target_metrics:
                if metric_key in layer_data:
                    metric_value = layer_data[metric_key]
                    processed_value = None
                    # --- Processing Logic (Consistent with extract_metric_vs_ckpt_data) ---
                    if isinstance(metric_value, np.ndarray):
                        if metric_value.size > 0:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                if metric_key == 'rmse':
                                    processed_value = float(np.sqrt(np.nanmean(metric_value**2)))
                                elif metric_key == 'val_loss_mean' or metric_key == 'val_loss': # Handle potential array loss
                                    processed_value = float(np.nanmean(metric_value))
                                # Add other array metrics if needed (e.g., dist, r2)
                                elif metric_key in ['dist', 'r2']:
                                     processed_value = float(np.nanmean(metric_value))
                                else: # Default to mean
                                     processed_value = float(np.nanmean(metric_value))
                        # else: skip empty array
                    elif isinstance(metric_value, (int, float, np.number)): # Handle scalars
                        processed_value = float(metric_value)
                    # else: skip other types

                    if processed_value is not None and np.isfinite(processed_value):
                        layer_metrics[layer_name][metric_key] = processed_value

    if not layer_metrics: return None # Return None if no metrics found for any layer
    return dict(layer_metrics) # Convert back to standard dict

def get_belief_dimension(checkpoint_data: dict, target_layer: str = 'combined') -> int:
    """ Attempts to find the belief dimension from predicted_beliefs in the target layer first, then others. """
    if not isinstance(checkpoint_data, dict): return 1 # Default dimension

    # Check target layer first
    target_layer_data = checkpoint_data.get(target_layer) # Use .get for safety
    if isinstance(target_layer_data, dict):
        pred_beliefs = target_layer_data.get('predicted_beliefs')
        if isinstance(pred_beliefs, np.ndarray) and pred_beliefs.ndim >= 2: # Check it's array and has >= 2 dims
                return pred_beliefs.shape[-1] # Return last dimension size

    # If not found in target layer, check other layers
    for layer, data in checkpoint_data.items():
        if isinstance(data, dict):
            pred_beliefs = data.get('predicted_beliefs')
            if isinstance(pred_beliefs, np.ndarray) and pred_beliefs.ndim >= 2:
               return pred_beliefs.shape[-1]

    # print(f"Warning: Could not determine belief dimension for target '{target_layer}'. Defaulting to 1.") # Less verbose
    return 1 # Default if not found anywhere

# --- Refactored Plotting Functions ---

def plot_belief_visualizations_on_axes(
    axes: list[plt.Axes],
    run_dir: str,
    is_markov3_run: bool,
    target_layer: str = "combined",
    num_panels_to_plot: int = 5,
    # Optional: Pass pre-loaded data if available to avoid reloading
    gt_data: dict | None = None,
    ckpt_indices_for_rmse: np.ndarray | None = None,
    rmse_values_for_rmse: np.ndarray | None = None,
):
    """ Plots belief progression (preds colored by GT) on the provided axes. """
    if len(axes) != num_panels_to_plot:
        print(f"Error: Number of axes ({len(axes)}) must match num_panels_to_plot ({num_panels_to_plot})")
        return None # Return None to indicate failure/no PCA

    experiment_name = "Markov3" if is_markov3_run else "QSlice"

    # --- 1. Select Checkpoints (Use provided or extract RMSE data) ---
    viz_ckpts_to_plot_str = []
    selected_indices_list = [] # Use list for final indices

    # If RMSE data isn't provided, extract it using the (now modified) extract_metric_vs_ckpt_data
    if ckpt_indices_for_rmse is None or rmse_values_for_rmse is None:
        print(f"--- Extracting RMSE data for {experiment_name} (for viz selection) ---")
        ckpt_indices_for_rmse, rmse_values_for_rmse = extract_metric_vs_ckpt_data(run_dir, is_markov3_run, target_layer, 'rmse')

    if ckpt_indices_for_rmse is None or len(ckpt_indices_for_rmse) < 1:
        print("Error: Could not extract/find any checkpoint indices. Cannot select checkpoints for visualization.")
        # Mark axes as N/A
        for ax in axes:
             ax.text(0.5, 0.5, "No Ckpts Found", ha='center', va='center', transform=ax.transAxes, fontsize=7, color='red')
             ax.set_axis_off()
        return None

    # --- 2. Select Checkpoints using Index Percentiles ---
    rmse_ckpts = ckpt_indices_for_rmse
    selected_indices = set()
    num_ckpts = len(rmse_ckpts)

    if num_ckpts < 1:
        print("Error: No checkpoints found to select from.")
        selected_indices_list = []
    elif num_ckpts <= num_panels_to_plot:
        # If fewer checkpoints exist than panels, just use all of them
        selected_indices = set(range(num_ckpts))
        print(f"Warning: Fewer than {num_panels_to_plot} checkpoints available ({num_ckpts}). Using all available.")
    else:
        # Select based on index percentiles
        idx_initial = 0
        idx_final = num_ckpts - 1
        # Calculate percentile indices (ensure they are valid indices)
        idx_p25 = min(max(0, int(0.01 * idx_final)), idx_final)
        idx_p75 = min(max(0, int(0.1 * idx_final)), idx_final)
        idx_p90 = min(max(0, int(0.35 * idx_final)), idx_final)

        selected_indices.add(idx_initial)
        selected_indices.add(idx_p25)
        selected_indices.add(idx_p75)
        selected_indices.add(idx_p90)
        selected_indices.add(idx_final)

        # If duplicates resulted in < num_panels_to_plot points, add more (e.g., 50% point)
        if len(selected_indices) < num_panels_to_plot:
            idx_p50 = min(max(0, int(0.50 * idx_final)), idx_final)
            selected_indices.add(idx_p50)
            # Add more points systematically if still needed (e.g. 10%, 40%, etc.)
            if len(selected_indices) < num_panels_to_plot:
                 idx_p10 = min(max(0, int(0.10 * idx_final)), idx_final)
                 selected_indices.add(idx_p10)
            # Add more logic here if needed to guarantee num_panels_to_plot unique points

    # Ensure final list is sorted and contains at most num_panels_to_plot indices
    final_selected_indices = sorted(list(selected_indices))
    if len(final_selected_indices) > num_panels_to_plot: # Should only happen if fallback logic adds too many
        # Prioritize keeping ends and roughly middle percentiles
        priority_indices = {idx_initial, idx_final, final_selected_indices[len(final_selected_indices)//2]}
        remaining = sorted(list(set(final_selected_indices) - priority_indices))
        needed = num_panels_to_plot - len(priority_indices)
        if needed > 0 and remaining:
             priority_indices.update(remaining[:needed])
        final_selected_indices = sorted(list(priority_indices))[:num_panels_to_plot]

    selected_indices_list = final_selected_indices
    # Convert selected indices back to checkpoint strings
    viz_ckpts_to_plot_str = [str(rmse_ckpts[idx]) for idx in selected_indices_list]
    num_panels_actually_plotting = len(viz_ckpts_to_plot_str)
    print(f"{num_panels_actually_plotting} Checkpoints selected for visualization: {viz_ckpts_to_plot_str}")

    # --- 2. Load GT Data (Use provided or load) ---
    if gt_data is None:
        gt_filename = 'markov3_ground_truth_data.joblib' if is_markov3_run else 'ground_truth_data.joblib'
        print("--- Loading Ground Truth Data ---")
        gt_data = load_ground_truth(run_dir, filename=gt_filename)

    # Check GT data validity *after* loading or receiving it
    if not gt_data or not isinstance(gt_data, dict) or \
       gt_data.get('beliefs') is None or gt_data.get('probs') is None or \
       not isinstance(gt_data['beliefs'], np.ndarray) or not isinstance(gt_data['probs'], np.ndarray) or \
       gt_data['beliefs'].size == 0 or gt_data['probs'].size == 0:
        print("Error: Cannot proceed without valid ground truth data (dict with non-empty 'beliefs' and 'probs' arrays).")
        for i, ax in enumerate(axes):
             title = f"Ckpt: {viz_ckpts_to_plot_str[i]}" if i < len(viz_ckpts_to_plot_str) else "Error"
             ax.set_title(title, fontsize=9, pad=3)
             ax.text(0.5, 0.5, "GT Data Error", ha='center', va='center', transform=ax.transAxes, fontsize=7, color='red')
             ax.set_axis_off()
        return None

    # --- 3. Load Prediction Data for Selected Checkpoints ---
    # *** Use the load_specific_checkpoint_data helper here for robust loading ***
    pred_data = {} # ckpt_str -> pred_beliefs
    print("--- Loading Prediction Data for Selected Checkpoints (using helper) ---")
    for ckpt_str in viz_ckpts_to_plot_str:
        ckpt_layer_data = load_specific_checkpoint_data(run_dir, is_markov3_run, ckpt_str)
        # Check structure carefully using the validated data from the helper
        if isinstance(ckpt_layer_data, dict) and target_layer in ckpt_layer_data and \
           isinstance(ckpt_layer_data[target_layer], dict) and \
           'predicted_beliefs' in ckpt_layer_data[target_layer]:
             pred_beliefs_value = ckpt_layer_data[target_layer]['predicted_beliefs']
             # Ensure it's a numpy array and not empty (helper should have done this)
             if isinstance(pred_beliefs_value, np.ndarray) and pred_beliefs_value.size > 0:
                 pred_data[ckpt_str] = pred_beliefs_value
             else:
                 # This case might indicate an issue in the helper or original data
                 print(f"Warning: 'predicted_beliefs' from helper for layer '{target_layer}' at ckpt {ckpt_str} is not a valid non-empty array ({type(pred_beliefs_value)}).")
                 pred_data[ckpt_str] = None # Mark as missing/invalid
        else:
             print(f"Warning: Could not load valid prediction data structure using helper for layer '{target_layer}' at checkpoint {ckpt_str}")
             pred_data[ckpt_str] = None # Mark as missing

    # --- 4. Calculate Coordinates and Colors (once for GT) ---
    gt_beliefs = gt_data['beliefs']
    weights = gt_data['probs']
    params = _get_plotting_params(experiment_name)
    belief_dims = gt_beliefs.shape[1] if gt_beliefs.ndim == 2 else 1
    pca_instance = None # Initialize PCA instance

    # Calculate GT coordinates (needed for color calculation and potentially PCA fitting)
    print("--- Calculating Plot Coordinates & Colors ---")
    x_gt, y_gt, pca_instance = _calculate_plot_coords(
        gt_beliefs, gt_beliefs, params['use_pca'], params['project_to_simplex'], params['inds_to_plot'], pca_instance
    )

    # Calculate colors using original RGB scheme based on GT coords/dims
    colors_rgba = None
    if x_gt is not None and y_gt is not None and weights is not None:
        def normalize_dim_color(data_dim):
            # Handle potential NaNs before min/max
            valid_data = data_dim[np.isfinite(data_dim)]
            if valid_data.size == 0: return np.ones_like(data_dim) * 0.5 # All NaN or empty
            min_val, max_val = np.min(valid_data), np.max(valid_data)
            if max_val > min_val:
                 norm = (data_dim - min_val) / (max_val - min_val)
            else: # Handle case where all valid values are the same
                 norm = np.ones_like(data_dim) * 0.5
            return np.nan_to_num(norm, nan=0.5) # Convert any remaining NaNs to grey

        R = normalize_dim_color(x_gt)
        G = normalize_dim_color(y_gt)

        # Determine Blue channel source
        B_source = None
        if not params['use_pca'] and not params['project_to_simplex'] and belief_dims > 2:
             plotted_inds = params['inds_to_plot']
             # Find first index not already used for R or G (ensure it's within bounds)
             available_inds = [i for i in range(belief_dims) if i not in plotted_inds[:2]]
             if available_inds:
                 third_dim_index = available_inds[0]
                 B_source = gt_beliefs[:, third_dim_index]
             else: # Fallback if only 2D available after filtering
                 B_source = np.sqrt(x_gt**2 + y_gt**2) # Use magnitude as fallback
        else: # Use magnitude if PCA, simplex, or 2D
             B_source = np.sqrt(x_gt**2 + y_gt**2)

        B = normalize_dim_color(B_source)
        alpha_values = transform_for_alpha(weights, min_alpha=params['min_alpha'], transformation=params['transformation'])

        # Ensure all components are valid numpy arrays with the same shape before stacking
        num_points_expected = weights.shape[0]
        if all(isinstance(c, np.ndarray) and c.shape == (num_points_expected,) for c in [R, G, B, alpha_values]):
            colors_rgba = np.stack([R, G, B, alpha_values], axis=-1)
        else:
            print(f"Warning: Color component mismatch or error. Shapes: R={R.shape if isinstance(R,np.ndarray) else 'None'}, G={G.shape if isinstance(G,np.ndarray) else 'None'}, B={B.shape if isinstance(B,np.ndarray) else 'None'}, Alpha={alpha_values.shape if isinstance(alpha_values,np.ndarray) else 'None'}. Expected ({num_points_expected},). Using default colors.")
            colors_rgba = np.array([[0.5, 0.5, 0.5, 0.5]] * num_points_expected) # Default grey
    else:
        print("Warning: Could not calculate GT coordinates or weights invalid. Using default colors.")
        num_points_expected = gt_beliefs.shape[0] if gt_beliefs is not None else 0
        if num_points_expected > 0:
             colors_rgba = np.array([[0.5, 0.5, 0.5, 0.5]] * num_points_expected) # Default grey
        else:
             colors_rgba = np.empty((0,4)) # Empty array if no points

    # --- 5. Plot Each Panel (Predictions only, colored by GT) ---
    print("--- Plotting Visualization Panels ---")
    overall_min_x, overall_max_x = np.inf, -np.inf
    overall_min_y, overall_max_y = np.inf, -np.inf

    # First pass to find combined limits across all panels that have valid data
    valid_coords_found = False
    for i, ckpt_str in enumerate(viz_ckpts_to_plot_str):
        pred_beliefs = pred_data.get(ckpt_str) # Can be None
        # Calculate prediction coordinates using the fitted PCA instance (if any)
        x_pred, y_pred, _ = _calculate_plot_coords(
            pred_beliefs, gt_beliefs, params['use_pca'], params['project_to_simplex'], params['inds_to_plot'], pca_instance
        )
        # Update overall limits using both GT and valid Pred data for this panel
        panel_x = []; panel_y = []
        # Include GT coords only if they are valid
        if x_gt is not None and np.isfinite(x_gt).any(): panel_x.append(x_gt)
        if y_gt is not None and np.isfinite(y_gt).any(): panel_y.append(y_gt)
        # Include Pred coords only if they are valid
        if x_pred is not None and np.isfinite(x_pred).any(): panel_x.append(x_pred)
        if y_pred is not None and np.isfinite(y_pred).any(): panel_y.append(y_pred)

        if panel_x:
            full_x = np.concatenate([arr for arr in panel_x if arr is not None and arr.size > 0])
            finite_x = full_x[np.isfinite(full_x)]
            if finite_x.size > 0:
                overall_min_x = min(overall_min_x, np.min(finite_x))
                overall_max_x = max(overall_max_x, np.max(finite_x))
                valid_coords_found = True
        if panel_y:
             full_y = np.concatenate([arr for arr in panel_y if arr is not None and arr.size > 0])
             finite_y = full_y[np.isfinite(full_y)]
             if finite_y.size > 0:
                overall_min_y = min(overall_min_y, np.min(finite_y))
                overall_max_y = max(overall_max_y, np.max(finite_y))
                valid_coords_found = True

    # Set default limits if no valid coordinates were found at all
    if not valid_coords_found:
         print("Warning: No valid coordinates found across any visualization panel. Using default limits [0,1].")
         overall_min_x, overall_max_x = 0, 1
         overall_min_y, overall_max_y = 0, 1

    # Add padding to limits, handle cases where min/max are the same
    x_range = overall_max_x - overall_min_x
    y_range = overall_max_y - overall_min_y
    x_pad = x_range * 0.05 if x_range > 1e-6 else 0.05 # Add min padding if range is tiny
    y_pad = y_range * 0.05 if y_range > 1e-6 else 0.05
    final_xlim = (overall_min_x - x_pad, overall_max_x + x_pad)
    final_ylim = (overall_min_y - y_pad, overall_max_y + y_pad)


    # Second pass to plot with consistent limits
    for i, ckpt_str in enumerate(viz_ckpts_to_plot_str):
        if i >= len(axes): break # Should not happen with initial check
        ax = axes[i]
        ax.set_title(f"Ckpt: {ckpt_str}", fontsize=9, pad=3) # Set title first

        pred_beliefs = pred_data.get(ckpt_str) # Can be None

        # Calculate prediction coordinates again
        x_pred, y_pred, _ = _calculate_plot_coords(
            pred_beliefs, gt_beliefs, params['use_pca'], params['project_to_simplex'], params['inds_to_plot'], pca_instance
        )

        # Plot ONLY Predictions, colored by GT position, using prediction point size
        # Pass colors_rgba which should match gt_beliefs length
        _plot_beliefs_on_ax(ax, x_pred, y_pred, colors_rgba, params['point_size']['pred'])

        # Apply the calculated consistent limits
        ax.set_xlim(final_xlim)
        ax.set_ylim(final_ylim)
        # Aspect ratio is set in _plot_beliefs_on_ax

    # Handle axes for which no checkpoint was plotted (if num_available < num_panels)
    for i in range(num_panels_actually_plotting, len(axes)):
         axes[i].set_title("N/A", fontsize=9, pad=3)
         axes[i].text(0.5, 0.5, "No Ckpt Data", ha='center', va='center', transform=axes[i].transAxes, fontsize=7, color='grey')
         axes[i].set_axis_off()

    return pca_instance # Return PCA instance if it was used

def plot_rmse_over_training_on_ax(ax: plt.Axes, df: pd.DataFrame, runs_to_plot_config: list, target_layer: str):
    """
    Plots Normalized RMSE vs Checkpoint Index for multiple runs on the provided axis.
    Returns handles and labels for legend creation.
    """
    print(f"\n--- Plotting RMSE vs Training Progress ({target_layer} layer) ---")
    all_data_found = False
    min_x, max_x = float('inf'), float('-inf')
    handles = [] # Initialize list for legend handles
    labels = []  # Initialize list for legend labels

    # Filter DataFrame for the target layer
    layer_df = df[df['Layer Name (Mapped)'] == target_layer].copy()

    if layer_df.empty:
        print(f"Warning: No data found for layer '{target_layer}' in DataFrame for RMSE vs Training plot.")
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
        ax.set_xlabel("Training Progress (Tokens Seen)")
        ax.set_ylabel("Normalized RMSE")
        ax.set_title(f"RMSE vs Training ({target_layer})")
        return None, None # Return None if no data

    print(f"Plotting {len(runs_to_plot_config)} runs...")
    for run_config in runs_to_plot_config:
        label = run_config['label']
        run_df = layer_df[layer_df['Run Label'] == label].sort_values('Checkpoint')

        if not run_df.empty and 'Normalized RMSE' in run_df.columns and run_df['Normalized RMSE'].notna().any():
            all_data_found = True
            # Drop rows where essential data for this plot is missing
            plot_df = run_df.dropna(subset=['Checkpoint', 'Normalized RMSE'])
            ckpts = plot_df['Checkpoint']
            norm_rmse = plot_df['Normalized RMSE']

            if len(ckpts) > 0:
                min_x = min(min_x, ckpts.min())
                max_x = max(max_x, ckpts.max())

                style = run_plot_styles.get(label, {}) # Get style from dict
                color = style.get('color', '#000000')
                linestyle = style.get('linestyle', '-')
                linewidth = style.get('lw', 1.5)

                 # Extract dimension for legend (use the first available dimension for the run/layer)
                dimension = plot_df['Dimension'].iloc[0] if 'Dimension' in plot_df.columns and not plot_df.empty else '?'
                legend_label = f"{label}"

                # Plot the data - lines only
                line, = ax.plot(
                    ckpts, norm_rmse,
                    linestyle=linestyle,
                    color=color,
                    linewidth=linewidth,
                    label=legend_label,
                    zorder=10
                )
                # Store handles/labels for potential shared legend
                handles.append(line)
                labels.append(legend_label)
        else:
             print(f"  -> No plottable data found for {label} in layer '{target_layer}'.")

    # --- Customize Plot ---
    if all_data_found:
        #ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1) # Reference line
        ax.set_xlabel("Tokens Seen During Training", fontsize=11)
        ax.set_ylabel(f"Normalized RMSE", fontsize=11)
        ax.set_title(f"RMSE vs. Training Progress", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax.tick_params(axis='both', which='major', length=4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6)) # Adjust nbins as needed

        # Use scientific notation for large x-axis values if needed
        if max_x > 1e6: # Adjust threshold as needed
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        else:
             ax.ticklabel_format(style='plain', axis='x') # Use plain numbers otherwise

        # Set reasonable y-limits, ensuring 1.0 is visible
        min_y_data = layer_df['Normalized RMSE'].dropna().min()
        max_y_data = layer_df['Normalized RMSE'].dropna().max()
        if pd.notna(min_y_data) and pd.notna(max_y_data):
             y_pad = (max_y_data - min_y_data) * 0.05 if max_y_data > min_y_data else 0.1
             # Ensure y=1 is within limits, provide some padding
             plot_min_y = max(0, min_y_data - y_pad)
             plot_max_y = 1.01#max(1.0 + y_pad, max_y_data + y_pad) # Ensure 1.0 is visible
             ax.set_ylim(bottom=plot_min_y, top=plot_max_y)
        else:
             ax.set_ylim(bottom=0) # Default if limits invalid

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # *** RETURN handles and labels ***
        return handles, labels

    else:
        print(f"No data plotted for RMSE vs Training ({target_layer}).")
        ax.text(0.5, 0.5, "No Data Plotted", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='orange')
        # *** RETURN None if nothing plotted ***
        return None, None

def plot_rmse_vs_loss_on_ax(ax: plt.Axes, df: pd.DataFrame, target_layer: str):
    """Creates the styled plot for Normalized RMSE vs Validation Loss on the provided axis."""
    print(f"\n--- Plotting RMSE vs Validation Loss ({target_layer} layer) ---")

    # Filter the DataFrame for the specified layer
    layer_df = df[df['Layer Name (Mapped)'] == target_layer].copy()

    if layer_df.empty:
        print(f"Error: No data found for layer '{target_layer}' for RMSE vs Loss plot.")
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
        ax.set_xlabel("Validation Loss")
        # ax.set_ylabel("Normalized RMSE")  # Removed y-axis label
        ax.set_title(f"RMSE vs Loss ({target_layer})")
        return
    if 'val_loss_mean' not in layer_df.columns or layer_df['val_loss_mean'].isna().all():
        print(f"Error: 'val_loss_mean' column missing or all NaN in layer '{target_layer}'. Cannot generate RMSE vs Loss plot.")
        ax.text(0.5, 0.5, "Loss Data Missing", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
        ax.set_xlabel("Validation Loss")
        # ax.set_ylabel("Normalized RMSE")  # Removed y-axis label
        ax.set_title(f"RMSE vs Loss ({target_layer})")
        return
    if 'Normalized RMSE' not in layer_df.columns or layer_df['Normalized RMSE'].isna().all():
         print(f"Error: 'Normalized RMSE' column missing or all NaN in layer '{target_layer}'. Cannot generate RMSE vs Loss plot.")
         ax.text(0.5, 0.5, "RMSE Data Missing", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
         ax.set_xlabel("Validation Loss")
         # ax.set_ylabel("Normalized RMSE")  # Removed y-axis label
         ax.set_title(f"RMSE vs Loss ({target_layer})")
         return

    # Get unique run labels that actually have data for this layer
    run_labels = layer_df['Run Label'].unique()
    plotted_something = False

    print(f"Plotting {len(run_labels)} runs...")
    for label in run_labels:
        run_df = layer_df[layer_df['Run Label'] == label].sort_values('Checkpoint') # Sort by ckpt for line
        style = run_plot_styles.get(label, {}) # Get style from dict

        # Filter out NaN values for plotting AND required columns exist
        plot_run_df = run_df.dropna(subset=['val_loss_mean', 'Normalized RMSE'])

        if not plot_run_df.empty:
            plotted_something = True
            # Plot the original data points as markers
            ax.plot(
                plot_run_df['val_loss_mean'],
                plot_run_df['Normalized RMSE'],
                label=label,
                color=style.get('color', '#000000'),
                linestyle='',
                marker=style.get('marker', '.'), # Use marker from style
                markersize=3, # Smaller markers for line plot
                linewidth=0, # Slightly thinner line
                alpha=0.8
            )

    # --- Customize Plot ---
    if plotted_something:
        ax.set_xlabel("Validation Loss", fontsize=11)
        # ax.set_ylabel(f"Norm. RMSE", fontsize=11)  # Removed y-axis label
        ax.set_title(f"RMSE vs. Loss", fontsize=12)

        # Add legend (optional, might get crowded)
        # ax.legend(title="Run Type", fontsize=7, loc='best') # Small legend if needed

        #ax.axhline(y=1.0, color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.8, zorder=1)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax.tick_params(axis='both', which='major', direction='in', length=4)
        # ax.tick_params(axis='both', which='minor', direction='in', length=2)
        # ax.minorticks_on()

        # Y-axis limits (consistent with RMSE vs Training)
        min_y_data = layer_df['Normalized RMSE'].dropna().min()
        max_y_data = layer_df['Normalized RMSE'].dropna().max()
        if pd.notna(min_y_data) and pd.notna(max_y_data):
             y_pad = (max_y_data - min_y_data) * 0.05 if max_y_data > min_y_data else 0.1
             plot_min_y = max(0, min_y_data - y_pad)
             plot_max_y = 1.01#max(1.0 + y_pad, max_y_data + y_pad)
             ax.set_ylim(bottom=plot_min_y, top=plot_max_y)
        else:
             ax.set_ylim(bottom=0)

        # X-axis: Focus on the region where loss changes typically occur, possibly log scale
        min_loss_data = layer_df['val_loss_mean'].dropna().min()
        max_loss_data = layer_df['val_loss_mean'].dropna().max()

        if pd.notna(min_loss_data) and pd.notna(max_loss_data):
             # Always use symlog scale for x-axis
             try:
                 # Use symlog scale which handles both positive and negative values
                 #ax.set_xscale('symlog', linthresh=1.0005, linscale=7.)
    
                 
                 # Calculate appropriate padding for the left side
                 x_pad_left = 0.00001
                 
                 # Find a reasonable right limit that excludes extreme outliers
                 # Use the 95th percentile instead of the maximum to exclude extreme outliers
                 right_limit = 1.004
                 
                 # Set limits with padding on left, capped on right
                 ax.set_xlim(left=min_loss_data - x_pad_left, right=right_limit)
                 
                 # Set appropriate linear threshold for symlog scale
                 linear_threshold = min(abs(min_loss_data) / 10, 1.0) if min_loss_data != 0 else 0.01
                 
                 # Use matplotlib's ticker module for proper log locators
                 from matplotlib import ticker
                 
                 # Set specific minor ticks as requested
                 #minor_ticks = [1.0, 1.00025, 1.0005, 1.00075]
                 
                 # Add log-distributed tick between 1.0005 and 1.006
                 # Using logarithmic distribution for the last segment
                 log_tick = 1.0005 * (1.006/1.0005)**(1/3)
                 log_tick2 = 1.0005 * (1.006/1.0005)**(2/3)
                 #minor_ticks.extend([log_tick]) #log_tick2])
                 
                 #ax.set_xticks(minor_ticks, minor=True)
                 
                 # Add more major ticks for better readability in this specific range
                 #ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
                 
                 # Format x-axis labels as normal numbers, not scientific notation
                 #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
             except Exception as e:
                 print(f"Warning: Could not set symlog scale for x-axis: {e}. Trying again with defaults.")
                 try:
                     # Try again with default parameters
                     ax.set_xscale('symlog')
                     
                     # Still try to limit the right side
                     right_limit = layer_df['val_loss_mean'].dropna().quantile(0.99)
                     ax.set_xlim(right=right_limit)
                     
                     # Format x-axis labels as normal numbers
                    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
                 except:
                     print("Error: Failed to set symlog scale despite attempts. Check your data.")
                     # Fallback to linear as last resort
                     ax.set_xscale('linear')
                     right_limit = layer_df['val_loss_mean'].dropna().quantile(0.95)
                     ax.set_xlim(left=min_loss_data, right=right_limit)
                     
                     # Format x-axis labels as normal numbers
                     #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        #ax.set_xticks([1, 1.00025, 1.0005, 1.003, 1.006])  # Set specific major tick locations
        #ax.set_xticklabels(['1', '1.00025', '1.0005', '1.003', '1.006'])  # Set tick labels as strings
        


        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelleft=False)

    else:
        print(f"No data plotted for RMSE vs Loss ({target_layer}).")
        ax.text(0.5, 0.5, "No Data Plotted", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='orange')

def plot_rmse_vs_layer_on_ax(ax: plt.Axes, df: pd.DataFrame, checkpoint_id: int | str, layer_order: list):
    """ Create a scatter/line plot of Normalized RMSE by Layer for a specific checkpoint on the provided axis. """
    print(f"\n--- Plotting RMSE vs Layer (Checkpoint: {checkpoint_id}) ---")

    try:
        checkpoint_id_int = int(checkpoint_id)
    except (ValueError, TypeError):
        print(f"Error: Invalid checkpoint_id '{checkpoint_id}'. Must be convertible to integer.")
        ax.text(0.5, 0.5, "Invalid Ckpt ID", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
        return

    # Filter the DataFrame for the specified checkpoint
    filtered_df = df[df['Checkpoint'] == checkpoint_id_int].copy()

    if filtered_df.empty:
        print(f"Error: No data found for Checkpoint {checkpoint_id_int} for RMSE vs Layer plot.")
        ax.text(0.5, 0.5, f"No Data Ckpt {checkpoint_id_int}", ha='center', va='center', transform=ax.transAxes, fontsize=7, color='red')
        ax.set_xlabel("Layer Name (Mapped)")
        #ax.set_ylabel("Normalized RMSE")
        ax.set_title(f"RMSE by Layer (Ckpt: {checkpoint_id_int})")
        return

    if 'Normalized RMSE' not in filtered_df.columns or filtered_df['Normalized RMSE'].isna().all():
         print(f"Error: 'Normalized RMSE' column missing or all NaN for Checkpoint {checkpoint_id_int}.")
         ax.text(0.5, 0.5, "RMSE Data Missing", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
         ax.set_xlabel("Layer Name (Mapped)")
         #ax.set_ylabel("Normalized RMSE")
         ax.set_title(f"RMSE by Layer (Ckpt: {checkpoint_id_int})")
         return

    # Create a categorical type with the specified layer ordering
    # Use only layers present in the filtered data for categories to avoid errors
    present_layers = filtered_df['Layer Name (Mapped)'].unique()
    actual_layer_order = [layer for layer in layer_order if layer in present_layers]
    if not actual_layer_order:
        print(f"Warning: None of the specified layer_order items found in data for checkpoint {checkpoint_id_int}. Using alphabetical sort.")
        actual_layer_order = sorted(present_layers) # Fallback sort

    filtered_df['Layer Order'] = pd.Categorical(
        filtered_df['Layer Name (Mapped)'],
        categories=actual_layer_order,
        ordered=True
    )

    # Sort by the custom layer order, handling potential missing categories gracefully
    sorted_df = filtered_df.sort_values('Layer Order').dropna(subset=['Layer Order'])

    # Get unique run labels present in this checkpoint's data
    run_labels = sorted_df['Run Label'].unique()
    plotted_something = False

    print(f"Plotting {len(run_labels)} runs...")
    for label in run_labels:
        # Ensure layer order is correct within each label's data for line plotting
        label_df = sorted_df[sorted_df['Run Label'] == label].sort_values('Layer Order').dropna(subset=['Normalized RMSE'])
        style = run_plot_styles.get(label, {}) # Get style from dict

        if not label_df.empty:
            plotted_something = True
            # Plot scatter points
            ax.scatter(
                label_df['Layer Name (Mapped)'],
                label_df['Normalized RMSE'],
                s=40, # Smaller points for less clutter
                alpha=0.7,
                marker=style.get('marker', 'o'),
                color=style.get('color', '#000000'),
                label=label
            )
            # Add lines connecting the points
            ax.plot(
                label_df['Layer Name (Mapped)'], # X is categorical layer name
                label_df['Normalized RMSE'],     # Y is the numeric RMSE
                color=style.get('color', '#000000'),
                alpha=0.5,
                linestyle=style.get('linestyle', '-'),
                linewidth=style.get('lw', 1.5) * 0.8, # Slightly thinner lines
                marker=None # Don't repeat marker in line
            )

    # --- Customize Plot ---
    if plotted_something:
        #ax.set_xlabel('Layer', fontsize=11) # Shorten label
        #ax.set_ylabel('Norm. RMSE', fontsize=11) # Abbreviate
        # Extract short checkpoint ID for title if too long
        ckpt_title_str = str(checkpoint_id_int)
        if len(ckpt_title_str) > 10: ckpt_title_str = f"{ckpt_title_str[:4]}..{ckpt_title_str[-4:]}"
        ax.set_title(f'RMSE by Layer', fontsize=12)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelleft=False)

        # Add legend (optional)
        # ax.legend(title='Run Label', fontsize=7, loc='best')

        # Reference line
        #ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # Set Y limits consistent with other RMSE plots
        min_y_data = sorted_df['Normalized RMSE'].dropna().min()
        max_y_data = sorted_df['Normalized RMSE'].dropna().max()
        if pd.notna(min_y_data) and pd.notna(max_y_data):
             y_pad = (max_y_data - min_y_data) * 0.05 if max_y_data > min_y_data else 0.1
             plot_min_y = max(0, min_y_data - y_pad)
             plot_max_y = 1.01#max(1.0 + y_pad, max_y_data + y_pad)
             ax.set_ylim(bottom=plot_min_y, top=plot_max_y)
        else:
             ax.set_ylim(bottom=0)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    else:
         print(f"No data plotted for RMSE vs Layer (Checkpoint {checkpoint_id_int}).")
         ax.text(0.5, 0.5, "No Data Plotted", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='orange')


def main():
    import argparse
    import sys
    from pathlib import Path
    
    # Add scripts directory to path for DataManager import
    scripts_dir = Path(__file__).parent / 'scripts'
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    
    try:
        from data_manager import DataManager
    except ImportError:
        print("Warning: DataManager not available. Using local data only.")
        DataManager = None
    
    parser = argparse.ArgumentParser(description="Generate Figure 3: Multi-Panel Analysis")
    parser.add_argument("--data-source", choices=['local', 'huggingface', 'auto'], default='auto',
                       help="Data source for analysis files")
    parser.add_argument("--data-dir", type=str, 
                       default="scripts/activation_analysis/run_predictions_RCOND_FINAL",
                       help="Local directory containing analysis files")
    parser.add_argument("--output-dir", type=str, default="Figs",
                       help="Output directory for plots")
    parser.add_argument("--checkpoint", type=str, default="last",
                       help="Target checkpoint ('last' or specific number)")
    parser.add_argument("--layer", type=str, default="combined",
                       help="Target layer for analysis")
    
    args = parser.parse_args()
    
    # Set up data directory based on source
    if DataManager is not None and args.data_source != 'local':
        try:
            dm = DataManager(source=args.data_source, data_dir=args.data_dir)
            
            # Extract model IDs from the run configuration for targeted downloads
            model_ids = []
            runs_to_process = [
                {"label": "Transformer (QSlice)", "sweep": "20241205175736", "run_id_int": 17, "is_markov3": False},
                {"label": "LSTM (QSlice)", "sweep": "20241121152808", "run_id_int": 49, "is_markov3": False},
                {"label": "Transformer (Classical)", "sweep": "20241205175736", "run_id_int": 17, "is_markov3": True},
                {"label": "LSTM (Classical)", "sweep": "20241121152808", "run_id_int": 49, "is_markov3": True},
            ]
            for run_info in runs_to_process:
                model_id = f"{run_info['sweep']}_{run_info['run_id_int']}"
                if model_id not in model_ids:  # Avoid duplicates
                    model_ids.append(model_id)
            
            analysis_dir = dm.get_analysis_data_dir(model_ids=model_ids)
            output_base_dir = str(analysis_dir)
            print(f"Using data from: {output_base_dir}")
        except Exception as e:
            print(f"Error setting up DataManager: {e}")
            print(f"Falling back to local data: {args.data_dir}")
            output_base_dir = args.data_dir
    else:
        output_base_dir = args.data_dir
        print(f"Using local data from: {output_base_dir}")

    # --- Configuration ---
    plot_output_dir = args.output_dir  # Directory to save the final figure
    final_figure_filename = "Fig3.png" # Output filename

    # Layer configuration
    layer_name_map = { # Map internal names to plottable names
        'blocks.0.hook_resid_pre': 'Embed', 'blocks.0.hook_resid_post': 'L1',
        'blocks.1.hook_resid_post': 'L2', 'blocks.2.hook_resid_post': 'L3',
        'blocks.3.hook_resid_post': 'L4', 'ln_final.hook_normalized': 'LN',
        'input': 'Embed', # Assuming RNN 'input' corresponds to Embedding
        'layer0': 'L1', 'layer1': 'L2',
        'layer2': 'L3', 'layer3': 'L4', # Assuming RNN layers map like this
        'combined': 'Concat'
    }
    # Define the desired order for the layer plot
    layer_order_for_plot = ['Embed', 'L1', 'L2', 'L3', 'L4', 'LN', 'Concat']
    target_layer_for_timeseries = 'combined' # Layer used for RMSE vs Training and RMSE vs Loss

    # Run configurations (Used for data loading and identifying lines in plots)
    runs_to_process = [
        {"label": "Transformer (QSlice)", "sweep": "20241205175736", "run_id_int": 17, "is_markov3": False},
        {"label": "LSTM (QSlice)", "sweep": "20241121152808", "run_id_int": 49, "is_markov3": False},
        {"label": "Transformer (Classical)", "sweep": "20241205175736", "run_id_int": 17, "is_markov3": True},
        {"label": "LSTM (Classical)", "sweep": "20241121152808", "run_id_int": 49, "is_markov3": True},
    ]

    # --- Configuration for Specific Plots ---
    # Select ONE run for the belief visualization panels (e.g., the first one)
    run_for_belief_viz = runs_to_process[0]
    num_belief_viz_panels = 5

    # Select ONE checkpoint for the RMSE vs Layer plot (e.g., the last checkpoint of the first run)
    target_run_dir_for_ckpt = os.path.join(output_base_dir, f"{run_for_belief_viz['sweep']}_{run_for_belief_viz['run_id_int']}")
    checkpoint_id_for_layer_plot = find_checkpoint_str(target_run_dir_for_ckpt, run_for_belief_viz['is_markov3'], first=False) # Find last checkpoint
    if checkpoint_id_for_layer_plot is None:
        print("Error: Could not find last checkpoint for the layer plot. Please set manually.")
        # Fallback: Manually set a known checkpoint ID string if find_checkpoint_str fails
        checkpoint_id_for_layer_plot = "4075724800" # Example manual ID - CHANGE IF NEEDED
        print(f"Using fallback checkpoint ID for layer plot: {checkpoint_id_for_layer_plot}")

    metrics_to_extract = ['rmse', 'val_loss_mean'] # Metrics needed for the DataFrame

    # --- 1. Data Extraction for DataFrame (Used by bottom row plots) ---
    print("\n--- STEP 1: Extracting Data for DataFrame ---")
    all_data_records = []
    initial_rmse_cache = {} # Cache initial RMSE: (run_label, layer_name) -> initial_rmse

    for run_info in runs_to_process:
        label = run_info["label"]
        run_dir = os.path.join(output_base_dir, f"{run_info['sweep']}_{run_info['run_id_int']}")
        is_mkv3 = run_info["is_markov3"]
        print(f"\nProcessing Run for DF: {label} (Dir: {run_dir})")

        # Find and load the *first* checkpoint to get initial RMSE and dimension
        first_ckpt_str = find_checkpoint_str(run_dir, is_mkv3, first=True)
        if not first_ckpt_str: print(f"  Warning: Could not find *first* checkpoint for {label}. Skipping initial RMSE calculation."); continue
        first_ckpt_data = load_specific_checkpoint_data(run_dir, is_mkv3, first_ckpt_str) # Use helper for consistency here
        if not first_ckpt_data: print(f"  Warning: Could not load *first* checkpoint data ({first_ckpt_str}) for {label}. Skipping initial RMSE."); continue

        initial_layer_metrics = extract_layer_metrics(first_ckpt_data, ['rmse']) # Only need RMSE initially
        dimension = get_belief_dimension(first_ckpt_data, target_layer=target_layer_for_timeseries) # Use target layer if possible
        if initial_layer_metrics:
            for layer_name, metrics_dict in initial_layer_metrics.items():
                 initial_rmse_cache[(label, layer_name)] = metrics_dict.get('rmse', np.nan)
        else:
            print(f"  Warning: Could not extract initial RMSE metrics from first checkpoint for {label}.")

        # Process all checkpoints for this run using extract_metric_vs_ckpt_data (which now uses direct load)
        print(f"  Extracting metrics using extract_metric_vs_ckpt_data for {label}...")
        for metric_to_get in metrics_to_extract:
             ckpts, metric_vals = extract_metric_vs_ckpt_data(run_dir, is_mkv3, target_layer_for_timeseries, metric_to_get)
             if ckpts is not None and metric_vals is not None:
                 for i, ckpt_idx_int in enumerate(ckpts):
                     # Find existing record or create new one
                     # This part is complex as we need to merge metrics from separate calls
                     # It's better to modify extract_layer_metrics or load once and extract all needed metrics
                     # Reverting to the previous approach of loading once per checkpoint and extracting all metrics
                     pass # Placeholder - will revert the loading logic below

    # --- REVERTED Data Extraction Logic ---
    # Load each checkpoint once and extract all necessary metrics
    all_data_records = [] # Reset records
    initial_rmse_cache = {} # Reset cache

    for run_info in runs_to_process:
        label = run_info["label"]
        run_dir = os.path.join(output_base_dir, f"{run_info['sweep']}_{run_info['run_id_int']}")
        is_mkv3 = run_info["is_markov3"]
        print(f"\nProcessing Run for DF: {label} (Dir: {run_dir})")

        # Find and load the *first* checkpoint to get initial RMSE and dimension
        first_ckpt_str = find_checkpoint_str(run_dir, is_mkv3, first=True)
        if not first_ckpt_str: print(f"  Warning: Could not find *first* checkpoint for {label}. Skipping initial RMSE calculation."); continue
        # Use the helper loader here for consistency and validation of the first file
        first_ckpt_data = load_specific_checkpoint_data(run_dir, is_mkv3, first_ckpt_str)
        if not first_ckpt_data: print(f"  Warning: Could not load *first* checkpoint data ({first_ckpt_str}) for {label}. Skipping initial RMSE."); continue

        initial_layer_metrics_for_run = extract_layer_metrics(first_ckpt_data, ['rmse']) # Only need RMSE initially
        dimension = get_belief_dimension(first_ckpt_data, target_layer=target_layer_for_timeseries) # Use target layer if possible
        if initial_layer_metrics_for_run:
            for layer_name, metrics_dict in initial_layer_metrics_for_run.items():
                 initial_rmse_cache[(label, layer_name)] = metrics_dict.get('rmse', np.nan)
        else:
            print(f"  Warning: Could not extract initial RMSE metrics from first checkpoint for {label}.")

        # Process all checkpoints for this run
        ckpt_pattern = f"markov3_checkpoint_*.joblib" if is_mkv3 else f"checkpoint_*.joblib"
        ckpt_files = sorted(glob.glob(os.path.join(run_dir, ckpt_pattern)))
        if not ckpt_files: print(f"  Warning: No checkpoint files found for {label}."); continue

        print(f"  Processing {len(ckpt_files)} checkpoints...")
        for f in ckpt_files:
            match = re.search(r'_(\d+)\.joblib$', os.path.basename(f))
            if not match: continue
            ckpt_idx_str = match.group(1)
            try: ckpt_idx_int = int(ckpt_idx_str)
            except ValueError: continue

            # Load the current checkpoint ONCE using the helper function
            current_ckpt_data = load_specific_checkpoint_data(run_dir, is_mkv3, ckpt_idx_str)
            if not current_ckpt_data:
                print(f"  Warning: Failed to load checkpoint {ckpt_idx_str} for {label}. Skipping.")
                continue

            # Extract ALL required metrics (RMSE and Loss) for *all* layers from this loaded data
            current_layer_metrics = extract_layer_metrics(current_ckpt_data, metrics_to_extract)
            if not current_layer_metrics:
                # print(f"  Debug: No metrics extracted from checkpoint {ckpt_idx_str} for {label}.")
                continue

            for layer_name, metrics_dict in current_layer_metrics.items():
                current_rmse = metrics_dict.get('rmse')
                current_loss = metrics_dict.get('val_loss_mean') # Get loss for this layer/ckpt
                # Retrieve cached initial RMSE
                initial_rmse = initial_rmse_cache.get((label, layer_name), np.nan)

                # Calculate Normalized RMSE (handle division by zero or NaN)
                normalized_rmse = np.nan
                if current_rmse is not None and not np.isnan(initial_rmse):
                    if initial_rmse > 1e-9: # Avoid division by near-zero
                        normalized_rmse = current_rmse / initial_rmse
                    elif abs(current_rmse) < 1e-9: # If both are zero/tiny, normalized value is 1
                         normalized_rmse = 1.0
                    # else: initial_rmse is zero but current is not -> leave as NaN

                mapped_layer_name = layer_name_map.get(layer_name, layer_name) # Map to consistent name

                all_data_records.append({
                    'Run Label': label,
                    'Checkpoint': ckpt_idx_int,
                    'Layer Name (Original)': layer_name,
                    'Layer Name (Mapped)': mapped_layer_name,
                    'Raw RMSE': current_rmse,
                    'Initial RMSE': initial_rmse,
                    'Dimension': dimension,
                    'Normalized RMSE': normalized_rmse,
                    'val_loss_mean': current_loss # Add loss to record
                })
    # --- End of REVERTED Data Extraction ---


    if not all_data_records:
        print("\nError: No data records were created from checkpoints. Cannot generate plots.")
        exit()

    print("\n--- Creating DataFrame ---")
    df = pd.DataFrame(all_data_records)
    df = df.sort_values(by=['Run Label', 'Layer Name (Mapped)', 'Checkpoint']).reset_index(drop=True)
    print("\nDataFrame Info:")
    # Increase display options for info()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
         df.info(verbose=True, show_counts=True) # More detailed info
    print("\nDataFrame Head:")
    print(df.head())
    print("\nChecking for NaN values in key columns:")
    print(df[['Normalized RMSE', 'val_loss_mean']].isnull().sum())


    # --- 2. Create Figure and Nested GridSpec Layout ---
    print("\n--- STEP 2: Creating Figure Layout (Nested GridSpec) ---")
    fig = plt.figure(figsize=(12, 5)) # Adjust figsize as needed, maybe taller

    # Outer GridSpec: 2 rows, 1 column. Controls overall row heights.
    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.], hspace=0.1) # Add hspace

    # Top Inner GridSpec (for visualizations): 1 row, 5 columns within outer[0]
    gs_top = gridspec.GridSpecFromSubplotSpec(1, num_belief_viz_panels, subplot_spec=gs_outer[0], wspace=0.1) # Minimal wspace

    # Bottom Inner GridSpec (for analysis plots): 1 row, 3 columns (ratio 2:1:1) within outer[1]
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], width_ratios=[2, 1, 1], wspace=0.1) # Use 3 columns with ratios

    # Create axes using inner GridSpecs
    ax_viz = [fig.add_subplot(gs_top[0, i]) for i in range(num_belief_viz_panels)]
    ax_train = fig.add_subplot(gs_bottom[0, 0]) # Takes first column (50% width due to width_ratios)
    ax_loss = fig.add_subplot(gs_bottom[0, 1], sharey=ax_train)   # Takes second column (25% width)
    ax_layer = fig.add_subplot(gs_bottom[0, 2], sharey=ax_train)  # Takes third column (25% width)


    # --- 3. Call Plotting Functions ---
    print("\n--- STEP 3: Populating Subplots ---")

    # Plot Belief Visualizations (Top Row)
    print(f"--- Generating Belief Visualizations for: {run_for_belief_viz['label']} ---")
    viz_run_dir = os.path.join(output_base_dir, f"{run_for_belief_viz['sweep']}_{run_for_belief_viz['run_id_int']}")
    _ = plot_belief_visualizations_on_axes(
        axes=ax_viz,
        run_dir=viz_run_dir,
        is_markov3_run=run_for_belief_viz['is_markov3'],
        target_layer=target_layer_for_timeseries, # Use the same target layer
        num_panels_to_plot=num_belief_viz_panels,
    )

    # Plot RMSE over Training (Bottom Left)
    # *** Capture returned handles/labels ***
    train_handles, train_labels = plot_rmse_over_training_on_ax(
        ax=ax_train,
        df=df,
        runs_to_plot_config=runs_to_process,
        target_layer='Concat'
    )

    # Plot RMSE vs Loss (Bottom Middle)
    plot_rmse_vs_loss_on_ax(
        ax=ax_loss,
        df=df,
        target_layer='Concat'
    )

    # Plot RMSE vs Layer (Bottom Right)
    plot_rmse_vs_layer_on_ax(
        ax=ax_layer,
        df=df,
        checkpoint_id=checkpoint_id_for_layer_plot, # Use the determined checkpoint ID
        layer_order=layer_order_for_plot
    )

    # --- 4. Final Figure Touches ---
    print("\n--- STEP 4: Finalizing Figure ---")
    #fig.suptitle("Transformer vs LSTM Performance Analysis (TomQA & Classical Tasks)", fontsize=16, y=0.99) # Adjust y slightly

    # Adjust layout - Use tight_layout on the figure, might work better with nested grids
    fig.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect for suptitle and legend space

    # Add a shared legend at the bottom
    # *** Use captured handles/labels ***
    if train_handles and train_labels:
         fig.legend(train_handles, train_labels, loc='lower center', bbox_to_anchor=(0.5, -0.075), ncol=min(4, len(train_handles)), fontsize=14)
    else:
         print("Warning: Could not retrieve handles/labels for shared legend.")


    # --- 5. Save and Show ---
    os.makedirs(plot_output_dir, exist_ok=True)
    # Configure figure for optimal Figma editing
    # Set text to outlines for better Figma compatibility
    mpl.rcParams['svg.fonttype'] = 'none'  # 'none' keeps text as text elements
    mpl.rcParams['pdf.fonttype'] = 42  # Type 42 (TrueType) for PDF export
    mpl.rcParams['svg.hashsalt'] = None  # Ensure reproducible SVG output
    
    # Save as high-res PNG
    full_output_path = os.path.join(plot_output_dir, final_figure_filename)
    os.makedirs(plot_output_dir, exist_ok=True)
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    print(f"\nFig3 saved to: {full_output_path}")
    
    # Save in vector format (SVG) 
    vector_output_path = os.path.join(plot_output_dir, final_figure_filename.replace('.png', '.svg'))
    plt.savefig(vector_output_path, format='svg', bbox_inches='tight')
    print(f"SVG version saved to: {vector_output_path}")

    plt.show()

    print("\n--- Combined Plotting Script Finished ---")


if __name__ == "__main__":
    main()