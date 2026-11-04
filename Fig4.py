# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd # Keep pandas for potential DataFrame input convenience
from sklearn.metrics.pairwise import cosine_similarity # Ensure this is imported
from scipy.stats import linregress # For calculating R-squared
import collections
import glob
import os
import joblib
import copy # Needed for copying colormaps if modifying them
from matplotlib.colors import LogNorm # For logarithmic color scaling
from matplotlib.lines import Line2D # For creating legend handles
import matplotlib # For colormap access

# =============================================================================
#  Your Existing Helper Functions (Keep these as they are)
# =============================================================================

def nested_dict_factory():
    """Returns a defaultdict that defaults to a regular dictionary."""
    return collections.defaultdict(dict)

def get_checkpoint_list(RUN_DIR, EXP_FOLDER, is_markov):
    if is_markov:
        prefix = 'markov3_checkpoint_'
    else:
        prefix = 'checkpoint_'
    ckpt_files = glob.glob(os.path.join(RUN_DIR, EXP_FOLDER, prefix + '*'))
    # sort by final number in filename
    ckpt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found for prefix '{prefix}' in {os.path.join(RUN_DIR, EXP_FOLDER)}")
    return ckpt_files

def ground_truth(RUN_DIR, EXP_FOLDER, is_markov):
    if is_markov:
        pattern = os.path.join(RUN_DIR, EXP_FOLDER, "markov3_ground_truth_data.joblib")
    else:
        pattern = os.path.join(RUN_DIR, EXP_FOLDER, "ground_truth_data.joblib")
    gt_files = glob.glob(pattern)
    if not gt_files:
        raise FileNotFoundError(f"No ground truth file found matching '{pattern}'")
    return joblib.load(gt_files[0])

def load_and_process_data(run_dir, exp_folder, is_markov):
    """
    Load and process data for a specific experiment. Handles potential errors.
    """
    try:
        # Get the final checkpoint file
        ckpt_files = get_checkpoint_list(run_dir, exp_folder, is_markov)
        final_ckpt_file = ckpt_files[-1]
        # print(f"{'Markov3' if is_markov else 'Non-Markov'} checkpoint:", final_ckpt_file) # Less verbose

        # Load residual beliefs from checkpoint
        ckpt_data = joblib.load(final_ckpt_file)
        if 'combined' not in ckpt_data or 'predicted_beliefs' not in ckpt_data['combined']:
             raise ValueError(f"Checkpoint file {final_ckpt_file} missing 'combined/predicted_beliefs' key.")
        resid = ckpt_data['combined']['predicted_beliefs']

        # Load ground truth data
        gt_data = ground_truth(run_dir, exp_folder, is_markov)
        if 'beliefs' not in gt_data or 'probs' not in gt_data:
             raise ValueError(f"Ground truth file for {exp_folder} missing 'beliefs' or 'probs' key.")
        gt_beliefs = gt_data['beliefs']
        probs = gt_data['probs'] # Keep probs if needed elsewhere, otherwise ignore

        # Ensure shapes match before processing
        if resid.shape[0] != gt_beliefs.shape[0]:
            raise ValueError(f"Shape mismatch: resid {resid.shape} vs gt_beliefs {gt_beliefs.shape} in {exp_folder}")

        # Mean subtract the data
        resid = resid# - np.mean(resid, axis=0)
        gt_beliefs = gt_beliefs# - np.mean(gt_beliefs, axis=0)

        return resid, gt_beliefs, probs

    except FileNotFoundError as e:
        print(f"Error loading data for {exp_folder} ({'Markov3' if is_markov else 'Non-Markov'}): {e}")
        return None, None, None # Return None to indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during data loading for {exp_folder} ({'Markov3' if is_markov else 'Non-Markov'}): {e}")
        return None, None, None # Return None


def calculate_cosine_similarities(data):
    """
    Calculate cosine similarities and extract upper triangle. Returns None if input is None.
    """
    if data is None:
        return None
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(data)
    # Extract upper triangle (excluding diagonal)
    return cosine_sim[np.triu_indices_from(cosine_sim, k=1)]

# =============================================================================
# Plotting Function for a SINGLE Subplot in the Grid (No Marginals)
# =============================================================================

def plot_single_similarity_subplot(ax_joint, # Removed marginal axes
                                   gt_markov, resid_markov,
                                   gt_nonmarkov, resid_nonmarkov,
                                   process_name, bins=50,
                                   show_title=False, # Control title visibility
                                   show_xlabel=False, show_ylabel=False,
                                   show_xticklabels=False, show_yticklabels=False):
    """
    Draws the 2D histogram and R^2 text onto provided axes. No marginals.
    Applies custom log normalization (bottom 10% density transparent).

    Args:
        ax_joint (matplotlib.axes.Axes): Axes for the central 2D histogram.
        gt_markov, resid_markov (np.array): Markov3 data.
        gt_nonmarkov, resid_nonmarkov (np.array): Full Generator data.
        process_name (str): Name of the process for the subplot title.
        bins (int): Number of bins for histograms.
        show_title (bool): Whether to show the title above this subplot.
        show_xlabel (bool): Whether to show the x-axis label on ax_joint.
        show_ylabel (bool): Whether to show the y-axis label on ax_joint.
        show_xticklabels (bool): Whether to show x-axis tick labels.
        show_yticklabels (bool): Whether to show y-axis tick labels.
    """
    # --- Style and Color Definitions ---
    color_markov = '#0072B2'    # Blue
    cmap_markov_base = matplotlib.colormaps['Blues'] # Use modern access
    color_nonmarkov = '#D55E00' # Vermillion/Orange
    cmap_nonmarkov_base = matplotlib.colormaps['Oranges'] # Use modern access
    # marginal_alpha = 0.35 # No longer needed

    # --- Create Colormaps with White Background ---
    cmap_markov = copy.copy(cmap_markov_base)
    cmap_nonmarkov = copy.copy(cmap_nonmarkov_base)
    cmap_markov.set_under('white', alpha=0) # Make values below threshold transparent
    cmap_nonmarkov.set_under('white', alpha=0)

    # --- Filter Data ---
    valid_markov = np.isfinite(gt_markov) & np.isfinite(resid_markov)
    valid_nonmarkov = np.isfinite(gt_nonmarkov) & np.isfinite(resid_nonmarkov)

    gt_markov_valid = gt_markov[valid_markov]
    resid_markov_valid = resid_markov[valid_markov]
    gt_nonmarkov_valid = gt_nonmarkov[valid_nonmarkov]
    resid_nonmarkov_valid = resid_nonmarkov[valid_nonmarkov]

    # --- Calculate R-squared ---
    r2_markov, r2_nonmarkov = np.nan, np.nan # Default to NaN
    if np.sum(valid_markov) > 1:
        _, _, r_value_m, _, _ = linregress(gt_markov_valid, resid_markov_valid)
        r2_markov = r_value_m**2
    if np.sum(valid_nonmarkov) > 1:
        _, _, r_value_nm, _, _ = linregress(gt_nonmarkov_valid, resid_nonmarkov_valid)
        r2_nonmarkov = r_value_nm**2

    # --- Central 2D Histograms (Custom Log Scale) ---
    # Determine common range for consistent binning
    all_gt = np.concatenate([gt_markov_valid, gt_nonmarkov_valid])
    all_resid = np.concatenate([resid_markov_valid, resid_nonmarkov_valid])

    if len(all_gt) == 0 or len(all_resid) == 0:
        print(f"Warning: No valid data points for {process_name}, skipping plot.")
        ax_joint.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax_joint.transAxes, fontsize=10) # Larger text
        if show_title:
             ax_joint.set_title(process_name, fontsize=12, pad=10) # Larger title, less padding needed
        ax_joint.tick_params(labelbottom=False, labelleft=False) # Hide ticks if no data
        # Turn off spines for empty plots for consistency
        for spine in ax_joint.spines.values():
            spine.set_visible(False)
        return

    xmin, xmax = (np.min(all_gt), np.max(all_gt)) if np.ptp(all_gt) > 0 else (np.min(all_gt)-0.1, np.max(all_gt)+0.1)
    ymin, ymax = (np.min(all_resid), np.max(all_resid)) if np.ptp(all_resid) > 0 else (np.min(all_resid)-0.1, np.max(all_resid)+0.1)
    hist_range = [[xmin, xmax], [ymin, ymax]]

    # Calculate histogram counts first to determine normalization range
    H_m, _, _ = np.histogram2d(gt_markov_valid, resid_markov_valid, bins=bins, range=hist_range)
    H_nm, _, _ = np.histogram2d(gt_nonmarkov_valid, resid_nonmarkov_valid, bins=bins, range=hist_range)

    # Find combined non-zero counts
    all_counts = np.concatenate([H_m[H_m > 0], H_nm[H_nm > 0]])

    if len(all_counts) > 0:
        # Calculate 10th percentile of non-zero counts as threshold
        vmin_thresh = np.percentile(all_counts, 10)
        vmax = np.max(all_counts)
        # Ensure vmin is positive for LogNorm, use a very small number if percentile is 0 or negative
        vmin_log = max(vmin_thresh, 1e-1)
        if vmin_log >= vmax: # Handle edge case where threshold >= max
             vmin_log = max(vmax * 0.01, 1e-1) # Use 1% of max or small number

        custom_norm = LogNorm(vmin=vmin_log, vmax=vmax, clip=True)
    else:
        # Default norm if no data or only zero counts
        custom_norm = LogNorm(vmin=0.1, vmax=1.0, clip=True)
        vmin_thresh = 0.1 # Set threshold for cmin consistency

    # Plotting parameters for hist2d
    # Use the calculated norm. cmin ensures bins exactly AT the threshold are also transparent.
    hist2d_args = {'bins': bins, 'range': hist_range, 'rasterized': True,
                   'norm': custom_norm, 'cmin': vmin_thresh + 1e-9} # Use norm, set cmin slightly above threshold

    # Plot Markov3 data
    ax_joint.hist2d(gt_markov_valid, resid_markov_valid, cmap=cmap_markov, **hist2d_args)
    # Plot Non-Markov data (overlay with alpha)
    ax_joint.hist2d(gt_nonmarkov_valid, resid_nonmarkov_valid, cmap=cmap_nonmarkov, alpha=0.75, **hist2d_args)

    # Labels and Grid for central plot only
    if show_xlabel:
        # Shorter X Label
        ax_joint.set_xlabel('True Belief Similarity', fontsize=11, labelpad=6)
    if show_ylabel:
        # Shorter Y Label
        ax_joint.set_ylabel('Predicted Similarity', fontsize=11, labelpad=6)
    ax_joint.grid(True, linestyle=':', alpha=0.3, color='grey') # Even more subtle grid

    # Control tick label visibility
    ax_joint.tick_params(axis='both', which='major', labelsize=10) # Larger
    ax_joint.tick_params(labelbottom=show_xticklabels, labelleft=show_yticklabels)

    # --- Add R-squared Text (Colored and Streamlined) ---
    # Position text box in bottom right corner
    # Create colored R² values
    props = dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor='#cccccc', linewidth=0.5)
    
    # Format R² values with colors
    r2_text_lines = []
    if not np.isnan(r2_markov):
        r2_text_lines.append(f"R² = {r2_markov:.2f}")
    else:
        r2_text_lines.append("R² = N/A")
    
    if not np.isnan(r2_nonmarkov):
        r2_text_lines.append(f"R² = {r2_nonmarkov:.2f}")
    else:
        r2_text_lines.append("R² = N/A")
    
    # Create text with different colors for each line
    text_y_start = 0.03
    for i, (text, color) in enumerate(zip(r2_text_lines, [color_markov, color_nonmarkov])):
        ax_joint.text(0.97, text_y_start + i*0.04, text, transform=ax_joint.transAxes, 
                      fontsize=9, color=color, weight='bold',
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=props if i == 0 else None) # Only add box to first line

    # --- Marginal Histograms Removed ---

    # --- Clean up Marginal Axes Removed ---

    # --- Subplot Title ---
    if show_title:
        ax_joint.set_title(process_name, fontsize=12, pad=10) # Larger title, less padding needed now


# =============================================================================
# Main Script Logic for Creating the Grid Figure (No Marginals)
# =============================================================================

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
    
    parser = argparse.ArgumentParser(description="Generate Figure 4: Representational Similarity Analysis")
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
            
            # Extract model IDs from experiment configurations for targeted downloads
            model_ids = []
            # Define experiment folders (same as used below)
            transformer_folders = {
                "Post-Quantum": "20250421221507_0", "MESS3": "20241205175736_23",
                "Bloch Walk": "20241205175736_17", "FRDN": "20250422023003_1"
            }
            lstm_folders = {
                "Post-Quantum": "20241121152808_48", "MESS3": "20241121152808_55",
                "Bloch Walk": "20241121152808_49", "FRDN": "20241121152808_53"
            }
            
            # Collect all unique model IDs
            all_folders = {**transformer_folders, **lstm_folders}
            for folder in all_folders.values():
                if folder not in model_ids:
                    model_ids.append(folder)
            
            analysis_dir = dm.get_analysis_data_dir(model_ids=model_ids, download_all_checkpoints=False)
            RUN_DIR = str(analysis_dir)
            print(f"Using data from: {RUN_DIR}")
        except Exception as e:
            print(f"Error setting up DataManager: {e}")
            print(f"Falling back to local data: {args.data_dir}")
            RUN_DIR = args.data_dir
    else:
        RUN_DIR = args.data_dir
        print(f"Using local data from: {RUN_DIR}")

    MAXSIZE = 1000 # Keep sampling for efficiency

    # Define experiment folders
    transformer_folders = {
        "Post-Quantum": "20250421221507_0", "MESS3": "20241205175736_23",
        "Bloch Walk": "20241205175736_17", "FRDN": "20250422023003_1"
    }
    lstm_folders = {
        "Post-Quantum": "20241121152808_48", "MESS3": "20241121152808_55",
        "Bloch Walk": "20241121152808_49", "FRDN": "20241121152808_53"
    }

    # Define the order of processes for columns - FRDN commented out, reordered
    # process_order = ["Post-Quantum", "MESS3", "Bloch Walk", "FRDN"]
    process_order = ["MESS3", "Bloch Walk", "Post-Quantum"]  # New order without FRDN

    # Process display names with type labels
    process_display_names = {
        "MESS3": "Mess3 Process\nClassical",
        "Bloch Walk": "Bloch Walk Process\nQuantum", 
        "Post-Quantum": "Moon Process\nPost-Quantum"
    }
    model_types = ["Transformer", "LSTM"]
    model_folders = {"Transformer": transformer_folders, "LSTM": lstm_folders}

    print(f'Starting processing and plotting grid (sampling to {MAXSIZE})...')

    # --- Create the main figure and GridSpec layout ---
    # Adjusted figure size for 3 columns instead of 4
    fig = plt.figure(figsize=(11, 7.5))  # Reduced width from 14 to 11

    # Outer grid - now 2x3 instead of 2x4
    outer_gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15) # 3 columns now

    # Use a consistent style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Store axes for sharing limits - now 2x3
    all_joint_axes = np.empty((2, 3), dtype=object)

    # --- Loop through models (rows) and processes (columns) ---
    for r, model_type in enumerate(model_types):
        print(f"\n--- Processing {model_type} Models ---")
        folders = model_folders[model_type]

        for c, process_name in enumerate(process_order):
            print(f"Processing: {model_type} - {process_name}")
            folder = folders.get(process_name)
            
            # Get display name for this process
            display_name = process_display_names.get(process_name, process_name)

            # --- Create Axes (No Inner GridSpec needed) ---
            ax_joint = fig.add_subplot(outer_gs[r, c])
            all_joint_axes[r, c] = ax_joint # Store axis

            if folder is None:
                print(f"  Folder not defined for {model_type} - {process_name}, skipping.")
                ax_joint.text(0.5, 0.5, 'No Folder', ha='center', va='center', transform=ax_joint.transAxes, color='gray', fontsize=10)
                # Only show title if it's the top row
                if r == 0:
                     ax_joint.set_title(display_name, fontsize=12, pad=10) # Use display name
                ax_joint.tick_params(labelbottom=False, labelleft=False)
                # Turn off spines for empty plots
                for spine in ax_joint.spines.values():
                    spine.set_visible(False)
                continue


            # --- Load Data ---
            resid_markov, gt_beliefs_markov, _ = load_and_process_data(RUN_DIR, folder, True)
            resid_nonmarkov, gt_beliefs_nonmarkov, _ = load_and_process_data(RUN_DIR, folder, False)

            if resid_markov is None or resid_nonmarkov is None:
                 print(f"  Skipping plot for {model_type} - {process_name} due to data loading errors.")
                 ax_joint.text(0.5, 0.5, 'Load Error', ha='center', va='center', transform=ax_joint.transAxes, color='red', fontsize=10)
                 if r == 0:
                     ax_joint.set_title(display_name, fontsize=12, pad=10) # Use display name
                 ax_joint.tick_params(labelbottom=False, labelleft=False)
                 for spine in ax_joint.spines.values():
                    spine.set_visible(False)
                 continue

            # --- Sampling ---
            if len(resid_markov) > MAXSIZE:
                # print(f'  Sampling {MAXSIZE} points from Markov3 data.') # Less verbose
                idx_m = np.random.choice(len(resid_markov), MAXSIZE, replace=False)
                resid_markov, gt_beliefs_markov = resid_markov[idx_m], gt_beliefs_markov[idx_m]
            if len(resid_nonmarkov) > MAXSIZE:
                # print(f'  Sampling {MAXSIZE} points from Non-Markov data.') # Less verbose
                idx_nm = np.random.choice(len(resid_nonmarkov), MAXSIZE, replace=False)
                resid_nonmarkov, gt_beliefs_nonmarkov = resid_nonmarkov[idx_nm], gt_beliefs_nonmarkov[idx_nm]

            # --- Calculate Cosine Similarities ---
            gt_sim_m = calculate_cosine_similarities(gt_beliefs_markov)
            res_sim_m = calculate_cosine_similarities(resid_markov)
            gt_sim_nm = calculate_cosine_similarities(gt_beliefs_nonmarkov)
            res_sim_nm = calculate_cosine_similarities(resid_nonmarkov)

            if gt_sim_m is None or res_sim_m is None or gt_sim_nm is None or res_sim_nm is None:
                print(f"  Skipping plot for {model_type} - {process_name} due to similarity calculation errors.")
                ax_joint.text(0.5, 0.5, 'Sim Error', ha='center', va='center', transform=ax_joint.transAxes, color='red', fontsize=10)
                if r == 0:
                    ax_joint.set_title(display_name, fontsize=12, pad=10) # Use display name
                ax_joint.tick_params(labelbottom=False, labelleft=False)
                for spine in ax_joint.spines.values():
                    spine.set_visible(False)
                continue

            # --- Plot on the created axes ---
            show_title = (r == 0) # Show title only for the top row (r=0)
            show_xlabel = (r == 1) # Show x-label only for the bottom row
            show_ylabel = (c == 0) # Show y-label only for the left column
            show_xticklabels = (r == 1) # Show x-ticks only for the bottom row
            show_yticklabels = (c == 0) # Show y-ticks only for the left column

            plot_single_similarity_subplot(ax_joint, # Pass only the main axis
                                           gt_sim_m, res_sim_m,
                                           gt_sim_nm, res_sim_nm,
                                           display_name,  # Use display name instead
                                           bins=50,
                                           show_title=show_title, # Control title visibility
                                           show_xlabel=show_xlabel,
                                           show_ylabel=show_ylabel,
                                           show_xticklabels=show_xticklabels,
                                           show_yticklabels=show_yticklabels)

    # --- Share x axes within columns only ---
    # Process each column separately
    for c in range(all_joint_axes.shape[1]):  # Iterate through columns
        # Find column-specific x limits
        col_xmin, col_xmax = np.inf, -np.inf
        col_has_valid_axes = False
        
        # First pass: determine column-specific x limits
        for r in range(all_joint_axes.shape[0]):  # Iterate through rows in this column
            ax = all_joint_axes[r, c]
            if ax is not None and (ax.collections or ax.lines):
                col_has_valid_axes = True
                xmin, xmax = ax.get_xlim()
                col_xmin = min(col_xmin, xmin)
                col_xmax = max(col_xmax, xmax)
        
        # Second pass: apply column-specific x limits
        if col_has_valid_axes and np.isfinite(col_xmin):
            print(f"Column {c}: Applying shared x limits: [{col_xmin:.2f}, {col_xmax:.2f}]")
            for r in range(all_joint_axes.shape[0]):
                ax = all_joint_axes[r, c]
                if ax is not None and (ax.collections or ax.lines):
                    ax.set_xlim(col_xmin, col_xmax)
        elif col_has_valid_axes:
            print(f"Column {c}: No valid x limits found, not sharing x axis.")


    # --- Add Row Labels ---
    row_label_kwargs = dict(fontsize=16, fontweight='bold', rotation=90, ha='right', va='center') # Larger
    fig.text(0.02, 0.7, model_types[0], **row_label_kwargs) # Adjusted x pos slightly
    fig.text(0.02, 0.3, model_types[1], **row_label_kwargs) # Adjusted x pos slightly


    # --- Add Shared Legend Below Grid ---
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Classical Approximation\nMarkov-Order 3',
                              markerfacecolor='#0072B2', markersize=12), # Blue, Larger marker
                       Line2D([0], [0], marker='s', color='w', label='Full Generator',
                              markerfacecolor='#D55E00', markersize=12)] # Orange, Larger marker

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), # Moved down slightly for multi-line
               ncol=2, frameon=False, fontsize=12) # Slightly smaller for multi-line


    # --- Overall Figure Title ---
    fig.suptitle('Representational Similarity Analysis', fontsize=22, y=0.98, weight='light') # More elegant title

    # --- Final Adjustments ---
    # Adjust spacing - adjusted for legend and beautification
    fig.subplots_adjust(left=0.08, bottom=0.17, right=0.97, top=0.91) # Adjusted for multi-line legend

    # Set style parameters before saving
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.linewidth'] = 0.8  # Thinner axes
    plt.rcParams['axes.edgecolor'] = '#333333'  # Darker gray axes

    # Use the output directory from args
    output_path = os.path.join(args.output_dir, "Fig4.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Fig4 saved to: {output_path}")
    
    # Also save as SVG
    svg_path = os.path.join(args.output_dir, "Fig4.svg")
    plt.savefig(svg_path, format="svg", bbox_inches='tight')
    print(f"SVG version saved to: {svg_path}")

    plt.show()

    print("\nGrid processing and plotting complete.")


if __name__ == "__main__":
    main()