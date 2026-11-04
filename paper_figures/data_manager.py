"""
Universal Data Manager for Epsilon-Transformers

Provides flexible data loading from either local directories or HuggingFace datasets.
Supports automatic download and caching for public reproducibility.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, List
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from tqdm.auto import tqdm
import warnings

class DataManager:
    """
    Universal data manager for loading analysis files and model checkpoints.
    
    Supports multiple data sources:
    - 'local': Use existing local directories
    - 'huggingface': Download from HuggingFace dataset
    - 'auto': Try local first, fallback to HuggingFace download
    """
    
    def __init__(
        self, 
        source: str = 'auto',
        data_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        repo_id: str = 'SimplexAI/quantum-representations'
    ):
        """
        Initialize DataManager.
        
        Args:
            source: Data source ('local', 'huggingface', 'auto')
            data_dir: Local directory to check for data
            cache_dir: Cache directory for HuggingFace downloads
            repo_id: HuggingFace repository ID
        """
        self.source = source
        self.repo_id = repo_id
        
        # Set up directories
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'epsilon-transformers'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None
            
        # Model mappings for validation
        self.model_mappings = {
            # Mess3 (Classical Process)
            "20241121152808_55": ("LSTM", "Mess3"),
            "20241205175736_23": ("Transformer", "Mess3"),
            "20241121152808_63": ("GRU", "Mess3"),
            "20241121152808_71": ("RNN", "Mess3"),
            
            # FRDN (Quantum Process)
            "20241121152808_53": ("LSTM", "FRDN"),
            "20250422023003_1": ("Transformer", "FRDN"),
            "20241121152808_61": ("GRU", "FRDN"),
            "20241121152808_69": ("RNN", "FRDN"),
            
            # Bloch Walk (Quantum Process)
            "20241121152808_49": ("LSTM", "Bloch Walk"),
            "20241205175736_17": ("Transformer", "Bloch Walk"),
            "20241121152808_57": ("GRU", "Bloch Walk"),
            "20241121152808_65": ("RNN", "Bloch Walk"),
            
            # Moon Process (Post-Quantum Process)
            "20241121152808_48": ("LSTM", "Moon Process"),
            "20250421221507_0": ("Transformer", "Moon Process"),
            "20241121152808_56": ("GRU", "Moon Process"),
            "20241121152808_64": ("RNN", "Moon Process"),
        }
    
    def get_analysis_data_dir(self, model_ids: List[str] = None, download_all_checkpoints: bool = True) -> Path:
        """Get directory containing analysis files."""
        if self.source == 'local':
            if self.data_dir is None:
                raise ValueError("data_dir must be provided for local source")
            return self._get_local_analysis_dir()
        elif self.source == 'huggingface':
            return self._get_hf_analysis_dir(model_ids, download_all_checkpoints=download_all_checkpoints)
        elif self.source == 'auto':
            # Try local first
            if self.data_dir is not None and self._check_local_analysis_exists():
                return self._get_local_analysis_dir()
            else:
                print("Local data not found, downloading from HuggingFace...")
                return self._get_hf_analysis_dir(model_ids, download_all_checkpoints=download_all_checkpoints)
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def get_models_data_dir(self, model_ids: List[str] = None) -> Path:
        """Get directory containing model checkpoints."""
        if self.source == 'local':
            if self.data_dir is None:
                raise ValueError("data_dir must be provided for local source")
            return self._get_local_models_dir()
        elif self.source == 'huggingface':
            return self._get_hf_models_dir(model_ids)
        elif self.source == 'auto':
            # Try local first
            if self.data_dir is not None and self._check_local_models_exists():
                return self._get_local_models_dir()
            else:
                print("Local models not found, downloading from HuggingFace...")
                return self._get_hf_models_dir(model_ids)
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def get_analysis_files(self, model_id: str) -> Dict[str, Path]:
        """
        Get analysis files for a specific model.
        
        Args:
            model_id: Model identifier (e.g., '20241121152808_57')
            
        Returns:
            Dictionary with analysis file paths
        """
        if model_id not in self.model_mappings:
            raise ValueError(f"Unknown model_id: {model_id}")
        
        analysis_dir = self.get_analysis_data_dir()
        model_dir = analysis_dir / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Analysis directory not found: {model_dir}")
        
        files = {
            'ground_truth': model_dir / 'ground_truth_data.joblib',
            'markov3_ground_truth': model_dir / 'markov3_ground_truth_data.joblib',
        }
        
        # Find checkpoint files
        checkpoint_files = list(model_dir.glob('checkpoint_*.joblib'))
        markov3_files = list(model_dir.glob('markov3_checkpoint_*.joblib'))
        
        if checkpoint_files:
            files['checkpoints'] = checkpoint_files
        if markov3_files:
            files['markov3_checkpoints'] = markov3_files
            
        return files
    
    def get_model_checkpoints(self, model_id: str) -> List[Path]:
        """
        Get model checkpoint files for a specific model.
        
        Args:
            model_id: Model identifier (e.g., '20241121152808_57')
            
        Returns:
            List of checkpoint file paths
        """
        if model_id not in self.model_mappings:
            raise ValueError(f"Unknown model_id: {model_id}")
        
        models_dir = self.get_models_data_dir([model_id])  # Pass model_id for selective download
        model_dir = models_dir / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        checkpoint_files = list(model_dir.glob('*.pt'))
        return sorted(checkpoint_files)
    
    def list_available_models(self) -> List[str]:
        """List all available model IDs."""
        return list(self.model_mappings.keys())
    
    def get_model_info(self, model_id: str) -> Dict[str, str]:
        """Get information about a specific model."""
        if model_id not in self.model_mappings:
            raise ValueError(f"Unknown model_id: {model_id}")
        
        architecture, process = self.model_mappings[model_id]
        return {
            'model_id': model_id,
            'architecture': architecture,
            'process': process,
            'description': f"{architecture} trained on {process}"
        }
    
    def download_from_huggingface(self, force_download: bool = False) -> Path:
        """
        Download the complete dataset from HuggingFace.
        
        Args:
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading dataset from {self.repo_id}...")
        
        dataset_dir = self.cache_dir / 'hf_dataset'
        
        if dataset_dir.exists() and not force_download:
            print(f"Dataset already cached at {dataset_dir}")
            return dataset_dir
        
        # Download the entire repository
        try:
            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type='dataset',
                cache_dir=str(self.cache_dir),
                local_dir=str(dataset_dir),
                local_dir_use_symlinks=False
            )
            print(f"Dataset downloaded to {dataset_dir}")
            return Path(downloaded_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    
    def get_analysis_data_for_figure(self, plot_config: List[Dict]) -> Path:
        """
        Get analysis data directory for figure generation with minimal downloads.
        
        Args:
            plot_config: List of plot configuration dictionaries with 'models' entries
            
        Returns:
            Path to analysis data directory
        """
        # Extract all unique model IDs from the plot configuration
        model_ids = set()
        for config in plot_config:
            for model_type, (sweep, run_id) in config.get('models', []):
                model_id = f"{sweep}_{run_id}"
                model_ids.add(model_id)
        
        model_ids = list(model_ids)
        print(f"Figure requires models: {model_ids}")
        
        return self.get_analysis_data_dir(model_ids)
    
    def download_selective_models(self, model_ids: List[str], force_download: bool = False, download_all_checkpoints: bool = True) -> Path:
        """
        Download only specific model checkpoints from HuggingFace.
        
        Args:
            model_ids: List of model identifiers to download
            force_download: Force re-download even if cached
            download_all_checkpoints: Download all available checkpoints (default: True)
            
        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading model checkpoints from {self.repo_id} for models: {model_ids}")
        
        # Disable individual HuggingFace progress bars
        disable_progress_bars()
        
        dataset_dir = self.cache_dir / 'hf_dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        models_dir = dataset_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # First pass: count total files to download
        total_files_to_download = 0
        download_queue = []  # List of (model_id, filename, remote_path, local_path)
        
        for model_id in model_ids:
            if model_id not in self.model_mappings:
                print(f"Warning: Unknown model_id {model_id}, skipping")
                continue
                
            model_dir = models_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all model checkpoint files for this model from HF
            try:
                from huggingface_hub import list_repo_files
                
                all_files = list_repo_files(self.repo_id, repo_type='dataset')
                model_files = [f for f in all_files if f.startswith(f"models/{model_id}/") and f.endswith('.pt')]
                
                # Extract checkpoint numbers
                checkpoint_nums = set()
                for f in model_files:
                    # Extract filename from models/model_id/checkpoint.pt
                    filename = f.split('/')[-1]
                    checkpoint_num = filename.replace('.pt', '')
                    if checkpoint_num.isdigit():
                        checkpoint_nums.add(checkpoint_num)
                
                if checkpoint_nums:
                    sorted_nums = sorted([int(x) for x in checkpoint_nums])
                    
                    if download_all_checkpoints:
                        # Download ALL checkpoints
                        checkpoints_to_download = [str(x) for x in sorted_nums]
                    else:
                        # Download first and last only
                        checkpoints_to_download = [str(sorted_nums[0])]  # First (0)
                        if len(sorted_nums) > 1:
                            checkpoints_to_download.append(str(sorted_nums[-1]))  # Last
                else:
                    checkpoints_to_download = ['0']  # Fallback
                    
            except Exception as e:
                print(f"  Could not list remote model files: {e}, using fallback checkpoints")
                checkpoints_to_download = ['0']
            
            # Add checkpoint files to download queue
            for ckpt_id in checkpoints_to_download:
                filename = f"{ckpt_id}.pt"
                local_path = model_dir / filename
                
                if not (local_path.exists() and local_path.stat().st_size > 0) or force_download:
                    remote_path = f"models/{model_id}/{filename}"
                    download_queue.append((model_id, filename, remote_path, local_path))
                    total_files_to_download += 1
            
            # Also download loss.csv if it exists
            loss_filename = "loss.csv"
            loss_local_path = model_dir / loss_filename
            if not (loss_local_path.exists() and loss_local_path.stat().st_size > 0) or force_download:
                loss_remote_path = f"models/{model_id}/{loss_filename}"
                download_queue.append((model_id, loss_filename, loss_remote_path, loss_local_path))
                total_files_to_download += 1
        
        # Second pass: download all queued files with unified progress bar
        if total_files_to_download > 0:
            print(f"Downloading {total_files_to_download} model checkpoint files...")
            
            with tqdm(total=total_files_to_download, desc="Downloading model checkpoints", unit="file") as pbar:
                downloaded_count = 0
                for model_id, filename, remote_path, local_path in download_queue:
                    try:
                        hf_hub_download(
                            repo_id=self.repo_id,
                            repo_type='dataset',
                            filename=remote_path,
                            local_dir=str(dataset_dir)
                        )
                        downloaded_count += 1
                        pbar.set_postfix_str(f"Downloaded {filename}")
                        pbar.update(1)
                    except Exception as e:
                        pbar.set_postfix_str(f"Skipped {filename} (not found)")
                        pbar.update(1)
                        print(f"  Warning: Could not download {filename}: {e}")
            
            print(f"Model download completed: {downloaded_count}/{total_files_to_download} files downloaded to {dataset_dir}")
        else:
            print(f"All model files already cached in {dataset_dir}")
            
        return dataset_dir

    def download_selective(self, model_ids: List[str], force_download: bool = False, download_all_checkpoints: bool = True) -> Path:
        """
        Download only specific model data from HuggingFace.
        
        Args:
            model_ids: List of model identifiers to download
            force_download: Force re-download even if cached
            download_all_checkpoints: Download all available checkpoints (default: True for complete analysis)
            
        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading selective data from {self.repo_id} for models: {model_ids}")
        
        # Disable individual HuggingFace progress bars to keep our unified progress bar clean
        disable_progress_bars()
        
        dataset_dir = self.cache_dir / 'hf_dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_dir = dataset_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # First pass: count total files to download
        total_files_to_download = 0
        download_queue = []  # List of (model_id, file_type, filename, remote_path, local_path)
        
        for model_id in model_ids:
            if model_id not in self.model_mappings:
                print(f"Warning: Unknown model_id {model_id}, skipping")
                continue
                
            model_dir = analysis_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check ground truth files
            for gt_file in ['ground_truth_data.joblib', 'markov3_ground_truth_data.joblib']:
                local_path = model_dir / gt_file
                if not (local_path.exists() and local_path.stat().st_size > 0) or force_download:
                    remote_path = f"analysis/{model_id}/{gt_file}"
                    download_queue.append((model_id, "ground_truth", gt_file, remote_path, local_path))
                    total_files_to_download += 1
            
            # Check checkpoint files (dynamically find latest)
            try:
                from huggingface_hub import list_repo_files
                
                # Get all checkpoint files for this model from HF
                all_files = list_repo_files(self.repo_id, repo_type='dataset')
                model_files = [f for f in all_files if f.startswith(f"analysis/{model_id}/")]
                
                # Extract checkpoint numbers for both regular and markov3 files
                checkpoint_nums = set()
                for f in model_files:
                    if 'checkpoint_' in f and f.endswith('.joblib'):
                        # Extract number from checkpoint_XXXXX.joblib or markov3_checkpoint_XXXXX.joblib
                        parts = f.split('checkpoint_')
                        if len(parts) > 1:
                            num_part = parts[1].replace('.joblib', '')
                            if num_part.isdigit():
                                checkpoint_nums.add(num_part)
                
                if checkpoint_nums:
                    sorted_nums = sorted([int(x) for x in checkpoint_nums])
                    
                    if download_all_checkpoints:
                        # Download ALL checkpoints
                        checkpoints_to_download = [str(x) for x in sorted_nums]
                    else:
                        # Original selective logic - get first, some middle ones, and last
                        checkpoints_to_download = [str(sorted_nums[0])]  # First (0)
                        
                        if len(sorted_nums) > 1:
                            checkpoints_to_download.append(str(sorted_nums[-1]))  # Last
                        if len(sorted_nums) > 5:
                            # Add some intermediate ones
                            mid_idx = len(sorted_nums) // 2
                            checkpoints_to_download.append(str(sorted_nums[mid_idx]))
                            if len(sorted_nums) > 10:
                                quarter_idx = len(sorted_nums) // 4
                                checkpoints_to_download.append(str(sorted_nums[quarter_idx]))
                else:
                    checkpoints_to_download = ['0']  # Fallback
                    
            except Exception as e:
                print(f"  Could not list remote files: {e}, using fallback checkpoints")
                checkpoints_to_download = ['0', '204800', '4075724800']
            
            # Add checkpoint files to download queue
            for ckpt_id in checkpoints_to_download:
                for prefix in ['checkpoint_', 'markov3_checkpoint_']:
                    filename = f"{prefix}{ckpt_id}.joblib"
                    local_path = model_dir / filename
                    
                    if not (local_path.exists() and local_path.stat().st_size > 0) or force_download:
                        remote_path = f"analysis/{model_id}/{filename}"
                        download_queue.append((model_id, "checkpoint", filename, remote_path, local_path))
                        total_files_to_download += 1
        
        # Second pass: download all queued files with unified progress bar
        if total_files_to_download > 0:
            print(f"Downloading {total_files_to_download} files...")
            
            with tqdm(total=total_files_to_download, desc="Downloading analysis files", unit="file") as pbar:
                downloaded_count = 0
                for model_id, file_type, filename, remote_path, local_path in download_queue:
                    try:
                        hf_hub_download(
                            repo_id=self.repo_id,
                            repo_type='dataset',
                            filename=remote_path,
                            local_dir=str(dataset_dir)
                        )
                        downloaded_count += 1
                        pbar.set_postfix_str(f"Downloaded {filename}")
                        pbar.update(1)
                    except Exception as e:
                        # Skip files that don't exist (e.g., some checkpoint combinations)
                        pbar.set_postfix_str(f"Skipped {filename} (not found)")
                        pbar.update(1)
                        if file_type == "ground_truth":  # Ground truth files should exist
                            print(f"  Warning: Could not download {filename}: {e}")
            
            print(f"Download completed: {downloaded_count}/{total_files_to_download} files downloaded to {dataset_dir}")
        else:
            print(f"All files already cached in {dataset_dir}")
            
        return dataset_dir
    
    # Private methods
    def _get_local_analysis_dir(self) -> Path:
        """Get local analysis directory."""
        if self.data_dir.name == 'analysis':
            return self.data_dir
        else:
            # Assume data_dir contains 'analysis' subdirectory
            analysis_dir = self.data_dir / 'analysis'
            if analysis_dir.exists():
                return analysis_dir
            else:
                # Assume data_dir IS the analysis directory
                return self.data_dir
    
    def _get_local_models_dir(self) -> Path:
        """Get local models directory."""
        if self.data_dir.name == 'models':
            return self.data_dir
        else:
            # Assume data_dir contains 'models' subdirectory
            models_dir = self.data_dir / 'models'
            if models_dir.exists():
                return models_dir
            else:
                raise FileNotFoundError(f"Models directory not found in {self.data_dir}")
    
    def _get_hf_analysis_dir(self, model_ids: List[str] = None, download_all_checkpoints: bool = True) -> Path:
        """Get HuggingFace analysis directory."""
        if model_ids:
            # Use selective download for specific models
            dataset_dir = self.download_selective(model_ids, download_all_checkpoints=download_all_checkpoints)
        else:
            # Fall back to full download
            dataset_dir = self.download_from_huggingface()
        return dataset_dir / 'analysis'
    
    def _get_hf_models_dir(self, model_ids: List[str] = None) -> Path:
        """Get HuggingFace models directory."""
        if model_ids:
            # Use selective download for specific models
            dataset_dir = self.download_selective_models(model_ids)
        else:
            # Fall back to full download
            dataset_dir = self.download_from_huggingface()
        return dataset_dir / 'models'
    
    def _check_local_analysis_exists(self) -> bool:
        """Check if local analysis data exists."""
        if self.data_dir is None:
            return False
        
        try:
            analysis_dir = self._get_local_analysis_dir()
            # Check for a few expected model directories
            sample_models = ['20241121152808_57', '20241205175736_23']
            for model_id in sample_models:
                model_dir = analysis_dir / model_id
                gt_file = model_dir / 'ground_truth_data.joblib'
                if model_dir.exists() and gt_file.exists() and gt_file.stat().st_size > 0:
                    return True
            return False
        except:
            return False
    
    def _check_local_models_exists(self) -> bool:
        """Check if local model data exists."""
        if self.data_dir is None:
            return False
        
        try:
            models_dir = self._get_local_models_dir()
            # Check for a few expected model directories
            sample_models = ['20241121152808_57', '20241205175736_23']
            for model_id in sample_models:
                model_dir = models_dir / model_id
                if model_dir.exists():
                    pt_files = [f for f in model_dir.glob('*.pt') if f.stat().st_size > 0]
                    if pt_files:
                        return True
            return False
        except:
            return False


# Convenience functions for common use cases
def get_analysis_data(
    model_id: str, 
    source: str = 'auto', 
    data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """
    Convenience function to get analysis files for a model.
    
    Args:
        model_id: Model identifier (e.g., '20241121152808_57')
        source: Data source ('local', 'huggingface', 'auto')
        data_dir: Local directory to check (for 'local' or 'auto' source)
        
    Returns:
        Dictionary with analysis file paths
    """
    dm = DataManager(source=source, data_dir=data_dir)
    return dm.get_analysis_files(model_id)


def list_models(
    source: str = 'auto', 
    data_dir: Optional[Union[str, Path]] = None
) -> List[Dict[str, str]]:
    """
    Convenience function to list all available models with their info.
    
    Args:
        source: Data source ('local', 'huggingface', 'auto')
        data_dir: Local directory to check (for 'local' or 'auto' source)
        
    Returns:
        List of model information dictionaries
    """
    dm = DataManager(source=source, data_dir=data_dir)
    models = []
    for model_id in dm.list_available_models():
        models.append(dm.get_model_info(model_id))
    return models


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Epsilon-Transformers Data Manager")
    parser.add_argument("--source", choices=['local', 'huggingface', 'auto'], default='auto',
                       help="Data source")
    parser.add_argument("--data-dir", type=str, help="Local data directory")
    parser.add_argument("--list-models", action='store_true', help="List available models")
    parser.add_argument("--download", action='store_true', help="Download from HuggingFace")
    
    args = parser.parse_args()
    
    dm = DataManager(source=args.source, data_dir=args.data_dir)
    
    if args.download:
        dm.download_from_huggingface()
    
    if args.list_models:
        print("Available models:")
        for model_info in list_models(source=args.source, data_dir=args.data_dir):
            print(f"  {model_info['model_id']}: {model_info['description']}")