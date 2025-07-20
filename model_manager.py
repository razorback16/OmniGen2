import time
import threading
import gc
import torch
from typing import Optional, Tuple
from datetime import datetime, timedelta

from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
try:
    from diffusers.hooks import apply_group_offloading
except ImportError:
    # Fallback for older diffusers versions
    apply_group_offloading = None


class ModelManager:
    """Manages lazy loading and automatic unloading of OmniGen2 models."""
    
    def __init__(self, args, idle_timeout_seconds: int = 3600):
        """
        Initialize the model manager.
        
        Args:
            args: Arguments containing model paths and configuration
            idle_timeout_seconds: Time in seconds before unloading idle model (default: 1 hour)
        """
        self.args = args
        self.idle_timeout_seconds = idle_timeout_seconds
        self.pipeline: Optional[OmniGen2Pipeline] = None
        self.accelerator: Optional[Accelerator] = None
        self.weight_dtype: Optional[torch.dtype] = None
        self.last_access_time: Optional[datetime] = None
        self._lock = threading.Lock()
        self._loading = False
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start the background thread that checks for idle timeout."""
        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                time.sleep(60)  # Check every minute
                with self._lock:
                    if (self.pipeline is not None and 
                        self.last_access_time is not None and
                        datetime.now() - self.last_access_time > timedelta(seconds=self.idle_timeout_seconds)):
                        print(f"Model idle for {self.idle_timeout_seconds} seconds. Unloading...")
                        self._unload_models()
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _load_models(self):
        """Load models to GPU. Should be called with lock held."""
        if self.pipeline is not None:
            return  # Already loaded
        
        print("Loading models to GPU...")
        start_time = time.time()
        
        # Initialize accelerator and dtype
        self.accelerator = Accelerator(mixed_precision=self.args.dtype if self.args.dtype != 'fp32' else 'no')
        self.weight_dtype = torch.float32
        if self.args.dtype == 'fp16':
            self.weight_dtype = torch.float16
        elif self.args.dtype == 'bf16':
            self.weight_dtype = torch.bfloat16
        
        # Load pipeline
        self.pipeline = OmniGen2Pipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=self.weight_dtype,
            trust_remote_code=True,
        )
        
        # Load transformer
        if self.args.transformer_path:
            print(f"Loading transformer from {self.args.transformer_path}")
            self.pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                self.args.transformer_path,
                torch_dtype=self.weight_dtype,
            )
        else:
            self.pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                self.args.model_path,
                subfolder="transformer",
                torch_dtype=self.weight_dtype,
            )
        
        # Load LoRA if specified
        if self.args.transformer_lora_path:
            print(f"Loading LoRA from {self.args.transformer_lora_path}")
            self.pipeline.load_lora_weights(self.args.transformer_lora_path)
        
        # Set up scheduler
        if self.args.scheduler == "dpmsolver":
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
            self.pipeline.scheduler = scheduler
        
        # Move to GPU with appropriate offloading strategy
        if self.args.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        elif self.args.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        elif self.args.enable_group_offload and apply_group_offloading is not None:
            apply_group_offloading(self.pipeline.transformer, onload_device=self.accelerator.device, 
                                 offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            apply_group_offloading(self.pipeline.mllm, onload_device=self.accelerator.device, 
                                 offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            apply_group_offloading(self.pipeline.vae, onload_device=self.accelerator.device, 
                                 offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        else:
            self.pipeline = self.pipeline.to(self.accelerator.device)
        
        load_time = time.time() - start_time
        print(f"Models loaded to GPU in {load_time:.2f} seconds")
        
        # Update access time
        self.last_access_time = datetime.now()
    
    def _unload_models(self):
        """Unload models from GPU. Should be called with lock held."""
        if self.pipeline is None:
            return  # Already unloaded
        
        print("Unloading models from GPU...")
        
        # Move models to CPU first to free GPU memory
        if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
            self.pipeline.transformer.cpu()
        if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
            self.pipeline.vae.cpu()
        if hasattr(self.pipeline, 'mllm') and self.pipeline.mllm is not None:
            self.pipeline.mllm.cpu()
        
        # Clear the pipeline
        self.pipeline = None
        self.accelerator = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Models unloaded from GPU")
    
    def get_pipeline(self) -> Tuple[OmniGen2Pipeline, Accelerator]:
        """
        Get the pipeline and accelerator, loading if necessary.
        
        Returns:
            Tuple of (pipeline, accelerator)
        """
        with self._lock:
            # Prevent multiple simultaneous loads
            while self._loading:
                self._lock.release()
                time.sleep(0.1)
                self._lock.acquire()
            
            if self.pipeline is None:
                self._loading = True
                try:
                    self._load_models()
                finally:
                    self._loading = False
            
            # Update access time
            self.last_access_time = datetime.now()
            
            return self.pipeline, self.accelerator
    
    def shutdown(self):
        """Clean shutdown of the model manager."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        with self._lock:
            self._unload_models()