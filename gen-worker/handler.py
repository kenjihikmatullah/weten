import runpod
import torch
import os
import base64
import logging
import traceback
from typing import Dict, Any
import numpy as np
import imageio
from supabase import create_client
import uuid
from datetime import datetime
import torch.backends.cudnn as cudnn
import gc
from contextlib import contextmanager
import subprocess
import sys
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def torch_gc():
    """Context manager to automatically handle GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

class WanVideoGenerator:
    def __init__(self):
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wan_repo_path = "/app/Wan2.1"
        self.models_path = "/app/models"
        
        # Enable cuDNN autotuner
        if self.device == "cuda":
            cudnn.benchmark = True
            cudnn.deterministic = False
            
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Updated model configurations for both 1.3B and 14B models
        self.model_configs = {
            "wan-2.1-1.3b": {
                "hf_name": "Wan-AI/Wan2.1-T2V-1.3B",
                "local_path": f"{self.models_path}/Wan2.1-T2V-1.3B",
                "task": "t2v-1.3B",
                "recommended_size": "480x854",
                "sample_shift": 8,
                "sample_guide_scale": 6,
                "memory_optimization": True,
                "min_gpu_memory_gb": 8,
                "description": "Faster, lighter model suitable for quick generation"
            },
            "wan-2.1-14b": {
                "hf_name": "Wan-AI/Wan2.1-T2V-14B",
                "local_path": f"{self.models_path}/Wan2.1-T2V-14B",
                "task": "t2v-14B",
                "recommended_size": "720x1280",
                "sample_shift": 8,
                "sample_guide_scale": 7.5,
                "memory_optimization": True,
                "min_gpu_memory_gb": 24,
                "description": "Higher quality model with better detail and coherence"
            }
        }
        
        # Initialize Wan2.1 repository
        self._setup_wan_repo()
        
    def _setup_wan_repo(self):
        """Setup Wan2.1 repository and dependencies"""
        try:
            if not os.path.exists(self.wan_repo_path):
                logger.info("Cloning Wan2.1 repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/Wan-Video/Wan2.1.git", 
                    self.wan_repo_path
                ], check=True)
                
            # Install Wan2.1 requirements
            requirements_path = os.path.join(self.wan_repo_path, "requirements.txt")
            if os.path.exists(requirements_path):
                logger.info("Installing Wan2.1 requirements...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", requirements_path
                ], check=True)
                
            # Add Wan2.1 to Python path
            if self.wan_repo_path not in sys.path:
                sys.path.insert(0, self.wan_repo_path)
                
            logger.info("Wan2.1 repository setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Wan2.1 repository: {str(e)}")
            raise e
    
    def check_gpu_compatibility(self, model_key: str) -> bool:
        """Check if current GPU has enough memory for the specified model"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return True
            
        model_config = self.model_configs.get(model_key)
        if not model_config:
            return False
            
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        required_memory_gb = model_config["min_gpu_memory_gb"]
        
        compatible = gpu_memory_gb >= required_memory_gb
        
        if not compatible:
            logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) insufficient for {model_key} "
                         f"(requires {required_memory_gb}GB)")
        else:
            logger.info(f"GPU memory check passed: {gpu_memory_gb:.1f}GB >= {required_memory_gb}GB")
            
        return compatible
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models and GPU compatibility"""
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
        model_info = {}
        for key, config in self.model_configs.items():
            compatible = self.check_gpu_compatibility(key)
            model_info[key] = {
                "description": config["description"],
                "recommended_size": config["recommended_size"],
                "min_gpu_memory_gb": config["min_gpu_memory_gb"],
                "compatible": compatible,
                "downloaded": os.path.exists(config["local_path"]) and bool(os.listdir(config["local_path"]))
            }
            
        return {
            "available_models": model_info,
            "current_gpu_memory_gb": gpu_memory_gb,
            "device": self.device
        }
    
    def download_model(self, model_key: str) -> bool:
        """Download the specified Wan model"""
        try:
            model_config = self.model_configs.get(model_key)
            if not model_config:
                raise ValueError(f"Unknown model: {model_key}")
            
            # Check GPU compatibility first
            if not self.check_gpu_compatibility(model_key):
                raise Exception(f"Insufficient GPU memory for {model_key}. "
                              f"Required: {model_config['min_gpu_memory_gb']}GB")
            
            local_path = model_config["local_path"]
            
            # Check if model already exists
            if os.path.exists(local_path) and os.listdir(local_path):
                logger.info(f"Model {model_key} already exists at {local_path}")
                return True
            
            # Create models directory
            os.makedirs(self.models_path, exist_ok=True)
            
            logger.info(f"Downloading model {model_key} from {model_config['hf_name']}...")
            logger.info(f"This may take a while for the 14B model...")
            
            # Download using huggingface-cli with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    subprocess.run([
                        "huggingface-cli", "download", 
                        model_config["hf_name"],
                        "--local-dir", local_path,
                        "--resume-download"  # Enable resume for large downloads
                    ], check=True, timeout=3600)  # 1 hour timeout
                    
                    logger.info(f"Model {model_key} downloaded successfully to {local_path}")
                    return True
                    
                except subprocess.TimeoutExpired:
                    logger.warning(f"Download attempt {attempt + 1} timed out")
                    if attempt < max_retries - 1:
                        logger.info("Retrying download...")
                        continue
                    else:
                        raise Exception("Download timed out after multiple attempts")
                        
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying download...")
                        continue
                    else:
                        raise e
            
        except Exception as e:
            logger.error(f"Error downloading model {model_key}: {str(e)}")
            return False
    
    def select_optimal_model(self, preferred_model: str = None) -> str:
        """Select the optimal model based on preferences and GPU compatibility"""
        if preferred_model and preferred_model in self.model_configs:
            if self.check_gpu_compatibility(preferred_model):
                return preferred_model
            else:
                logger.warning(f"Preferred model {preferred_model} not compatible, falling back...")
        
        # Try models in order of preference (14B first if compatible, then 1.3B)
        for model_key in ["wan-2.1-14b", "wan-2.1-1.3b"]:
            if self.check_gpu_compatibility(model_key):
                logger.info(f"Selected model: {model_key}")
                return model_key
        
        # Fallback to 1.3B if nothing else works
        logger.warning("Using 1.3B model as fallback")
        return "wan-2.1-1.3b"
    
    def generate_video(self, prompt: str, duration: float = 3.0, resolution: str = None, 
                      model: str = None, guidance_scale: float = None, 
                      return_method: str = "base64") -> Dict[str, Any]:
        """Generate video from text prompt using Wan2.1"""
        with torch_gc():
            try:
                # Select optimal model if not specified
                if not model:
                    model = self.select_optimal_model()
                elif model not in self.model_configs:
                    raise ValueError(f"Unknown model: {model}. Available: {list(self.model_configs.keys())}")
                
                # Check compatibility
                if not self.check_gpu_compatibility(model):
                    fallback_model = self.select_optimal_model()
                    logger.warning(f"Falling back to {fallback_model} due to GPU constraints")
                    model = fallback_model
                
                model_config = self.model_configs[model]
                
                # Download model if needed
                if not os.path.exists(model_config["local_path"]) or not os.listdir(model_config["local_path"]):
                    logger.info(f"Model not found locally, downloading {model}...")
                    if not self.download_model(model):
                        raise Exception(f"Failed to download model: {model}")
                
                # Use model-specific defaults if not provided
                if resolution is None:
                    size_str = model_config["recommended_size"]
                    width, height = map(int, size_str.split('x'))
                elif 'x' in resolution:
                    width, height = map(int, resolution.split('x'))
                    size_str = f"{width}x{height}"
                else:
                    size_str = model_config["recommended_size"]
                    width, height = map(int, size_str.split('*'))
                
                if guidance_scale is None:
                    guidance_scale = model_config["sample_guide_scale"]
                
                logger.info(f"Generating video with {model}: {prompt}")
                logger.info(f"Resolution: {size_str}, Duration: {duration}s, Guidance: {guidance_scale}")
                
                # Create output directory
                output_dir = f"/app/outputs/{uuid.uuid4().hex}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Prepare generation command
                generate_script = os.path.join(self.wan_repo_path, "generate.py")
                
                cmd = [
                    "python", generate_script,
                    "--task", model_config["task"],
                    "--size", size_str,
                    "--ckpt_dir", model_config["local_path"],
                    "--sample_shift", str(model_config["sample_shift"]),
                    "--sample_guide_scale", str(guidance_scale),
                    "--prompt", prompt,
                    "--output_dir", output_dir
                ]
                
                # Add memory optimization flags
                if model_config.get("memory_optimization", False):
                    cmd.extend(["--offload_model", "True"])
                    # For 14B model, add additional memory optimizations
                    if "14b" in model.lower():
                        cmd.extend(["--t5_cpu", "--low_vram"])
                
                # Change to Wan2.1 directory for execution
                original_cwd = os.getcwd()
                os.chdir(self.wan_repo_path)
                
                try:
                    logger.info(f"Running generation command: {' '.join(cmd)}")
                    # Longer timeout for 14B model
                    timeout = 1800 if "14b" in model.lower() else 600  # 30min for 14B, 10min for 1.3B
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
                    logger.info(f"Generation completed successfully")
                    logger.info(f"Generation output: {result.stdout}")
                    
                finally:
                    os.chdir(original_cwd)
                
                # Find generated video file
                video_file = None
                for file in os.listdir(output_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_file = os.path.join(output_dir, file)
                        break
                
                if not video_file or not os.path.exists(video_file):
                    raise Exception("Generated video file not found")
                
                # Return based on method
                if return_method == "base64":
                    with open(video_file, 'rb') as f:
                        video_bytes = f.read()
                    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                    
                    # Clean up
                    shutil.rmtree(output_dir, ignore_errors=True)
                    
                    return {
                        "video": video_base64,
                        "size_bytes": len(video_bytes),
                        "format": "base64",
                        "model_used": model
                    }
                    
                elif return_method == "file_info":
                    # Return file info for external upload
                    file_size = os.path.getsize(video_file)
                    
                    # Read file for upload
                    with open(video_file, 'rb') as f:
                        video_bytes = f.read()
                    
                    return {
                        "video_data": video_bytes,
                        "temp_path": video_file,
                        "output_dir": output_dir,
                        "size_bytes": file_size,
                        "format": "binary",
                        "model_used": model
                    }
                    
                else:
                    raise ValueError(f"Unknown return_method: {return_method}")
                    
            except subprocess.TimeoutExpired:
                error_msg = f"Video generation timed out for model {model}"
                logger.error(error_msg)
                raise Exception(error_msg)
            except subprocess.CalledProcessError as e:
                logger.error(f"Generation command failed: {e}")
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
                raise Exception(f"Video generation failed: {e.stderr}")
            except Exception as e:
                logger.error(f"Error generating video: {str(e)}")
                logger.error(traceback.format_exc())
                raise e

class SupabaseUploader:
    def __init__(self, url: str = None, key: str = None):
        self.supabase_url = url or os.getenv('SUPABASE_URL')
        self.supabase_key = key or os.getenv('SUPABASE_ANON_KEY')
        self.supabase = None
        
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase: {e}")
    
    def upload_video(self, video_data: bytes, filename: str = None, bucket: str = "videos") -> Dict[str, Any]:
        """Upload video to Supabase Storage"""
        try:
            if not self.supabase:
                raise Exception("Supabase not initialized")
            
            if not filename:
                filename = f"video_{uuid.uuid4().hex}_{int(datetime.now().timestamp())}.mp4"
            
            logger.info(f"Uploading video to Supabase: {filename}")
            
            # Upload to Supabase Storage
            result = self.supabase.storage.from_(bucket).upload(
                filename, 
                video_data,
                file_options={"content-type": "video/mp4"}
            )
            
            if result.error:
                raise Exception(f"Upload failed: {result.error}")
            
            # Get public URL
            public_url = self.supabase.storage.from_(bucket).get_public_url(filename)
            
            logger.info(f"Video uploaded successfully: {public_url}")
            
            return {
                "success": True,
                "filename": filename,
                "public_url": public_url,
                "size_bytes": len(video_data)
            }
            
        except Exception as e:
            logger.error(f"Error uploading to Supabase: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Global instances
video_generator = WanVideoGenerator()
supabase_uploader = SupabaseUploader()

def handler(job):
    """RunPod handler function"""
    try:
        # Get job input
        job_input = job.get("input", {})
        
        # Special endpoint to get model information
        if job_input.get("action") == "get_model_info":
            return video_generator.get_model_info()
        
        # Extract parameters
        prompt = job_input.get("prompt", "")
        duration = job_input.get("duration", 3.0)
        resolution = job_input.get("resolution")  # None = use model default
        model = job_input.get("model")  # None = auto-select optimal
        guidance_scale = job_input.get("guidance_scale")  # None = use model default
        
        # Output options
        output_method = job_input.get("output_method", "base64")  # base64, supabase, or both
        supabase_bucket = job_input.get("supabase_bucket", "videos")
        custom_filename = job_input.get("filename")
        
        # Validate inputs
        if not prompt:
            return {"error": "Prompt is required"}
        
        # Validate model if specified
        if model and model not in video_generator.model_configs:
            available_models = list(video_generator.model_configs.keys())
            return {"error": f"Invalid model: {model}. Available: {available_models}"}
        
        # Validate duration
        if duration <= 0 or duration > 30:
            return {"error": "Duration must be between 0 and 30 seconds"}
        
        # Validate output method
        valid_methods = ["base64", "supabase", "both"]
        if output_method not in valid_methods:
            return {"error": f"Invalid output_method: {output_method}. Valid: {valid_methods}"}
        
        logger.info(f"Processing job - Model: {model or 'auto-select'}, Prompt: {prompt[:50]}...")
        
        # Generate video
        if output_method in ["supabase", "both"]:
            # Generate with file info for upload
            result = video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                model=model,
                guidance_scale=guidance_scale,
                return_method="file_info"
            )
            
            response = {
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution or video_generator.model_configs[result["model_used"]]["recommended_size"],
                "model": result["model_used"],
                "size_bytes": result["size_bytes"],
                "status": "success"
            }
            
            # Upload to Supabase if requested
            if output_method in ["supabase", "both"]:
                upload_result = supabase_uploader.upload_video(
                    result["video_data"],
                    custom_filename,
                    supabase_bucket
                )
                
                if upload_result["success"]:
                    response["supabase"] = {
                        "url": upload_result["public_url"],
                        "filename": upload_result["filename"]
                    }
                else:
                    response["supabase_error"] = upload_result["error"]
            
            # Include base64 if requested
            if output_method == "both":
                video_base64 = base64.b64encode(result["video_data"]).decode('utf-8')
                response["video_base64"] = video_base64
            
            # Clean up temp files
            if "output_dir" in result and os.path.exists(result["output_dir"]):
                shutil.rmtree(result["output_dir"], ignore_errors=True)
                
        else:
            # Generate with base64 only
            result = video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                model=model,
                guidance_scale=guidance_scale,
                return_method="base64"
            )
            
            response = {
                "video": result["video"],
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution or video_generator.model_configs[result["model_used"]]["recommended_size"],
                "model": result["model_used"],
                "size_bytes": result["size_bytes"],
                "status": "success"
            }
        
        return response
        
    except Exception as e:
        error_msg = f"Error processing job: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    
    # Show GPU information and model compatibility
    try:
        model_info = video_generator.get_model_info()
        logger.info(f"GPU Memory: {model_info['current_gpu_memory_gb']:.1f}GB")
        logger.info("Model compatibility:")
        for model_name, info in model_info['available_models'].items():
            status = "✓" if info['compatible'] else "✗"
            logger.info(f"  {status} {model_name}: {info['description']}")
    except Exception as e:
        logger.warning(f"Could not check model compatibility: {e}")
    
    # Pre-download optimal model
    try:
        optimal_model = video_generator.select_optimal_model()
        logger.info(f"Pre-downloading optimal model: {optimal_model}")
        video_generator.download_model(optimal_model)
        logger.info("Model download completed successfully")
    except Exception as e:
        logger.warning(f"Could not pre-download model: {str(e)}")
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})