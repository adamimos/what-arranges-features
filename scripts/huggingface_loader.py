"""
HuggingFace Model Loader for Epsilon-Transformers

Drop-in replacement for S3ModelLoader that works with the public HuggingFace dataset.
Provides compatible interface for existing analysis scripts.
"""

import os
import yaml
import json
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from scripts.data_manager import DataManager
from epsilon_transformers.training.networks import create_RNN
from transformer_lens import HookedTransformer, HookedTransformerConfig

class HuggingFaceModelLoader:
    """
    HuggingFace-based model loader that provides S3ModelLoader-compatible interface.
    
    This class allows existing analysis scripts to work with the public HuggingFace dataset
    without requiring changes to the calling code.
    """
    
    def __init__(self, repo_id: str = 'SimplexAI/quantum-representations', cache_dir: Optional[str] = None):
        """
        Initialize HuggingFace model loader.
        
        Args:
            repo_id: HuggingFace repository ID
            cache_dir: Local cache directory for downloads
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.data_manager = DataManager(source='huggingface', cache_dir=cache_dir, repo_id=repo_id)
        
        # Model mappings compatible with existing analysis code
        self.model_mappings = {
            # Mess3 (Classical Process)
            ("20241121152808", "55"): ("LSTM", "Mess3"),
            ("20241205175736", "23"): ("Transformer", "Mess3"),
            ("20241121152808", "63"): ("GRU", "Mess3"),
            ("20241121152808", "71"): ("RNN", "Mess3"),
            
            # FRDN (Quantum Process)
            ("20241121152808", "53"): ("LSTM", "FRDN"),
            ("20250422023003", "1"): ("Transformer", "FRDN"),
            ("20241121152808", "61"): ("GRU", "FRDN"),
            ("20241121152808", "69"): ("RNN", "FRDN"),
            
            # Bloch Walk (Quantum Process)
            ("20241121152808", "49"): ("LSTM", "Bloch Walk"),
            ("20241205175736", "17"): ("Transformer", "Bloch Walk"),
            ("20241121152808", "57"): ("GRU", "Bloch Walk"),
            ("20241121152808", "65"): ("RNN", "Bloch Walk"),
            
            # Moon Process (Post-Quantum Process)
            ("20241121152808", "48"): ("LSTM", "Moon Process"),
            ("20250421221507", "0"): ("Transformer", "Moon Process"),
            ("20241121152808", "56"): ("GRU", "Moon Process"),
            ("20241121152808", "64"): ("RNN", "Moon Process"),
        }
        
        # Create reverse mapping for run lookup
        self.sweep_to_runs = {}
        for (sweep_id, run_id), (arch, process) in self.model_mappings.items():
            if sweep_id not in self.sweep_to_runs:
                self.sweep_to_runs[sweep_id] = []
            run_name = self._generate_run_name(sweep_id, run_id, arch, process)
            self.sweep_to_runs[sweep_id].append(run_name)
    
    def _generate_run_name(self, sweep_id: str, run_id: str, arch: str, process: str) -> str:
        """Generate run name compatible with S3ModelLoader format."""
        # Convert process names to match expected format
        process_map = {
            "Mess3": "mess3",
            "FRDN": "fanizza", 
            "Bloch Walk": "tom_quantum",
            "Moon Process": "post_quantum"
        }
        
        arch_map = {
            "LSTM": "LSTM",
            "GRU": "GRU", 
            "RNN": "RNN",
            "Transformer": "transformer"
        }
        
        process_key = process_map.get(process, process.lower())
        arch_key = arch_map.get(arch, arch.upper())
        
        if arch == "Transformer":
            # Transformer format: run_ID_L4_H4_DH16_DM64_process
            return f"run_{run_id}_L4_H4_DH16_DM64_{process_key}"
        else:
            # RNN format: run_ID_L4_H64_ARCH_uni_process  
            return f"run_{run_id}_L4_H64_{arch_key}_uni_{process_key}"
    
    def _get_model_id(self, sweep_id: str, run_name: str) -> str:
        """Extract model ID from sweep and run."""
        # Extract run number from run name
        run_id = run_name.split('_')[1]
        return f"{sweep_id}_{run_id}"
    
    def list_sweeps(self) -> List[str]:
        """List all available sweep IDs."""
        return list(self.sweep_to_runs.keys())
    
    def list_runs_in_sweep(self, sweep_id: str) -> List[str]:
        """List all runs in a sweep."""
        return self.sweep_to_runs.get(sweep_id, [])
    
    def list_checkpoints(self, sweep_id: str, run_name: str) -> List[str]:
        """
        List all checkpoint files for a run.
        
        Returns checkpoint paths in S3ModelLoader-compatible format.
        """
        model_id = self._get_model_id(sweep_id, run_name)
        
        try:
            # Trigger download of model checkpoints if needed
            models_dir = self.data_manager.get_models_data_dir()
            checkpoints = self.data_manager.get_model_checkpoints(model_id)
            
            # Convert to S3-compatible format
            checkpoint_paths = []
            for ckpt_path in checkpoints:
                # Format: sweep_id/run_name/checkpoint.pt
                relative_path = f"{sweep_id}/{run_name}/{ckpt_path.name}"
                checkpoint_paths.append(relative_path)
            
            return sorted(checkpoint_paths)
            
        except FileNotFoundError as e:
            print(f"Warning: No checkpoints found for {model_id}: {e}")
            return []
        except Exception as e:
            print(f"Error listing checkpoints for {model_id}: {e}")
            return []
    
    def load_checkpoint(self, sweep_id: str, run_name: str, checkpoint_path: str, device: str = 'cpu') -> tuple:
        """
        Load a model checkpoint.
        
        Args:
            sweep_id: Sweep identifier
            run_name: Run name
            checkpoint_path: Path to checkpoint file
            device: Device to load model onto ('cpu' or 'cuda')
            
        Returns:
            Tuple of (model, run_config)
        """
        model_id = self._get_model_id(sweep_id, run_name)
        
        # Get checkpoint file
        checkpoint_name = checkpoint_path.split('/')[-1]
        checkpoints = self.data_manager.get_model_checkpoints(model_id)
        
        checkpoint_file = None
        for ckpt in checkpoints:
            if ckpt.name == checkpoint_name:
                checkpoint_file = ckpt
                break
        
        if checkpoint_file is None:
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found for {model_id}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        # Extract model from checkpoint
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            model_state = checkpoint
        
        # Load run config (if available)
        run_config = self._load_run_config(model_id)
        
        # Create model from checkpoint
        model = self._create_model_from_checkpoint(model_state, run_config, device)
        
        return model, run_config
    
    def _load_run_config(self, model_id: str) -> Dict[str, Any]:
        """Load run configuration for a model."""
        try:
            models_dir = self.data_manager.get_models_data_dir()
            config_path = models_dir / model_id / 'run_config.yaml'
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Generate basic config from model_id
                return self._generate_default_config(model_id)
                
        except Exception as e:
            print(f"Warning: Could not load config for {model_id}: {e}")
            return self._generate_default_config(model_id)
    
    def _generate_default_config(self, model_id: str) -> Dict[str, Any]:
        """Generate default configuration based on model ID."""
        # Extract sweep and run from model_id
        sweep_id, run_id = model_id.split('_')
        
        # Get model info
        model_key = (sweep_id, run_id)
        if model_key in self.model_mappings:
            arch, process = self.model_mappings[model_key]
        else:
            arch, process = "Unknown", "Unknown"
        
        # Process type mapping
        process_map = {
            "Mess3": "mess3",
            "FRDN": "fanizza",
            "Bloch Walk": "tom_quantum", 
            "Moon Process": "post_quantum"
        }
        
        process_name = process_map.get(process, process.lower())
        
        # Default process parameters
        process_params = {
            "tom_quantum": {"alpha": 2.5, "beta": 0.3},
            "mess3": {"x": 0.15, "a": 0.6},
            "fanizza": {"alpha": 2.0, "lamb": 0.5},
            "post_quantum": {"alpha": 2.718281828459045, "beta": 0.5}  # defaults from function
        }
        
        config = {
            "model_config": {
                "model_type": arch.lower(),
                "n_layers": 4,
                "n_ctx": 8,
            },
            "process_config": {
                "name": process_name,  # Required by get_matrix_from_args
                **process_params.get(process_name, {})  # Add process-specific parameters
            },
            "sweep_id": sweep_id,
            "run_id": run_id,
            "architecture": arch,
            "process": process
        }
        
        # Architecture-specific configs
        if arch == "Transformer":
            config["model_config"].update({
                "n_heads": 4,
                "d_head": 16,
                "d_model": 64
            })
        else:
            # RNN models need additional configuration parameters
            config["model_config"].update({
                "hidden_size": 64,
                "direction": "uni",
                "rnn_type": arch,  # LSTM, GRU, or RNN
                "num_layers": 4,
                "dropout": 0.0,
                "bidirectional": False
            })
        
        return config
    
    def _create_model_from_checkpoint(self, model_state: Dict, run_config: Dict, device: str = 'cpu') -> torch.nn.Module:
        """Create model from checkpoint state and config."""
        architecture = run_config.get("architecture", "Unknown")
        
        if architecture == "Transformer":
            return self._create_transformer_model(model_state, run_config, device)
        else:
            # For RNN models (LSTM, GRU, RNN)
            return self._create_rnn_model(model_state, run_config, device)
    
    def _create_rnn_model(self, model_state: Dict, run_config: Dict, device: str = 'cpu') -> torch.nn.Module:
        """Create RNN model from checkpoint state and config."""
        try:
            # Infer vocab size from the output layer in the state dict
            output_layer_weight = model_state['output_layer.weight']
            vocab_size = output_layer_weight.size(0)  # First dimension is output size (vocab size)
            
            # Create RNN model based on config structure
            model = create_RNN(run_config, vocab_size, device)
            
            # Load state dict
            model.load_state_dict(model_state)
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to create RNN model: {e}")
    
    def _create_transformer_model(self, model_state: Dict, run_config: Dict, device: str = 'cpu') -> torch.nn.Module:
        """Create Transformer model from checkpoint state and config."""
        try:
            # Prepare model config
            model_config = run_config['model_config'].copy()
            
            # Handle dtype conversion like S3 loader
            if 'dtype' in model_config:
                if isinstance(model_config['dtype'], str):
                    model_config['dtype'] = getattr(torch, model_config['dtype'].split('.')[-1])
            else:
                model_config['dtype'] = torch.float32
                
            model_config['device'] = device
            
            # Create and load model
            hooked_config = HookedTransformerConfig(**model_config)
            model = HookedTransformer(hooked_config)
            model.load_state_dict(model_state)
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to create Transformer model: {e}")
    
    def load_loss_from_run(self, sweep_id: str, run_name: str) -> pd.DataFrame:
        """Load loss data for a run."""
        model_id = self._get_model_id(sweep_id, run_name)
        
        try:
            models_dir = self.data_manager.get_models_data_dir()
            loss_path = models_dir / model_id / 'loss.csv'
            
            if loss_path.exists():
                return pd.read_csv(loss_path)
            else:
                print(f"Warning: No loss data found for {model_id}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Warning: Could not load loss data for {model_id}: {e}")
            return pd.DataFrame()
    
    def load_run_configs(self, sweep_id: str) -> Dict[str, Dict]:
        """Load run configurations for all runs in a sweep."""
        configs = {}
        for run_name in self.list_runs_in_sweep(sweep_id):
            model_id = self._get_model_id(sweep_id, run_name)
            configs[run_name] = self._load_run_config(model_id)
        return configs
    
    # Additional methods for compatibility
    def check_if_process_data_exists(self, *args, **kwargs) -> bool:
        """Compatibility method - always return True for HF data."""
        return True
    
    def get_sweep_ind(self, sweep_id: str) -> int:
        """Get sweep index (for compatibility)."""
        sweeps = self.list_sweeps()
        try:
            return sweeps.index(sweep_id)
        except ValueError:
            return -1


# Convenience function for easy migration
def create_hf_loader(repo_id: str = 'SimplexAI/quantum-representations') -> HuggingFaceModelLoader:
    """Create HuggingFace model loader with default settings."""
    return HuggingFaceModelLoader(repo_id=repo_id)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="HuggingFace Model Loader Test")
    parser.add_argument("--repo-id", default="SimplexAI/quantum-representations", 
                       help="HuggingFace repository ID")
    parser.add_argument("--list-sweeps", action='store_true', help="List available sweeps")
    parser.add_argument("--list-runs", type=str, help="List runs in sweep")
    parser.add_argument("--test-load", action='store_true', help="Test loading a model")
    
    args = parser.parse_args()
    
    loader = HuggingFaceModelLoader(repo_id=args.repo_id)
    
    if args.list_sweeps:
        print("Available sweeps:")
        for sweep in loader.list_sweeps():
            print(f"  {sweep}")
    
    if args.list_runs:
        runs = loader.list_runs_in_sweep(args.list_runs)
        print(f"Runs in sweep {args.list_runs}:")
        for run in runs:
            print(f"  {run}")
    
    if args.test_load:
        # Test loading first available model
        sweeps = loader.list_sweeps()
        if sweeps:
            sweep_id = sweeps[0]
            runs = loader.list_runs_in_sweep(sweep_id)
            if runs:
                run_name = runs[0]
                checkpoints = loader.list_checkpoints(sweep_id, run_name)
                if checkpoints:
                    print(f"Testing load of {sweep_id}/{run_name}/{checkpoints[0]}")
                    try:
                        model, config = loader.load_checkpoint(sweep_id, run_name, checkpoints[0])
                        print(f"✅ Successfully loaded model with config keys: {list(config.keys())}")
                    except Exception as e:
                        print(f"❌ Error loading model: {e}")
                else:
                    print("No checkpoints found")
            else:
                print("No runs found")
        else:
            print("No sweeps found")