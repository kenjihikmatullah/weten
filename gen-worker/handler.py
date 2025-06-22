import runpod
import torch
import os
import logging
import traceback
from typing import Dict, Any
import uuid
import shutil
import sys
from supabase import create_client, Client
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = '/runpod-volume/models'

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class WanVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Detect network volume path
        self.models_path = DEFAULT_MODELS_PATH
        logger.info(f"Using models path: {self.models_path}")
        
        # Use official Diffusers model names
        self.model_configs = {
            "wan-2.1-1.3b": {
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "local_path": f"{self.models_path}/Wan2.1-T2V-1.3B-Diffusers",
                "recommended_size": {"width": 854, "height": 480},
                "flow_shift": 3.0,
                "min_gpu_memory_gb": 8
            },
            "wan-2.1-14b": {
                "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "local_path": f"{self.models_path}/Wan2.1-T2V-14B-Diffusers",
                "recommended_size": {"width": 1280, "height": 720},
                "flow_shift": 5.0,
                "min_gpu_memory_gb": 24
            }
        }
        
        self._verify_models()
        
        # Initialize pipelines
        self.pipelines = {}
    
    
    def _list_volume_contents(self):
        """
        Debug function to list volume contents
        """
        logger.info("=== Volume Detection Debug ===")
        
        # Check environment variables
        for key, value in os.environ.items():
            if "volume" in key.lower() or "runpod" in key.lower():
                logger.info(f"ENV: {key} = {value}")
        
        # Check the specific RunPod volume path
        runpod_paths = ["/runpod-volume", "/runpod-volume/models"]
        for path in runpod_paths:
            if os.path.exists(path):
                try:
                    contents = os.listdir(path)
                    logger.info(f"Contents of {path}: {contents}")
                except Exception as e:
                    logger.info(f"Cannot read {path}: {e}")
            else:
                logger.info(f"Path does not exist: {path}")
        
        # Check common paths
        paths_to_check = ["/", "/workspace", "/volume", "/mnt"]
        
        for path in paths_to_check:
            if os.path.exists(path):
                try:
                    contents = os.listdir(path)
                    logger.info(f"Contents of {path}: {contents}")
                    
                    # Check subdirectories
                    for item in contents[:10]:  # Limit to first 10 items
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path) and not item.startswith('.'):
                            try:
                                sub_contents = os.listdir(item_path)
                                logger.info(f"  {item}/: {sub_contents[:5]}...")  # First 5 items
                            except:
                                pass
                except Exception as e:
                    logger.info(f"Cannot read {path}: {e}")
        
        logger.info("=== End Debug ===")

    def upload_to_storage(self, file_path: str, reference_id: str) -> str:
        """Upload video to Supabase storage and return public URL"""
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Upload to storage bucket 'videos'
            storage_path = f"public/{reference_id}.mp4"
            response = supabase.storage.from_('videos').upload(
                storage_path,
                file_bytes,
                {"content-type": "video/mp4"}
            )
            
            # Get public URL
            public_url = supabase.storage.from_('videos').get_public_url(storage_path)
            logger.info(f"Video uploaded successfully: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload video: {str(e)}")
            raise

    def _verify_models(self):
        """Verify that models exist in the volume"""
        logger.info(f"Verifying models in path: {self.models_path}")
        
        # If models path doesn't exist, try to create it
        if not os.path.exists(self.models_path):
            try:
                os.makedirs(self.models_path, exist_ok=True)
                logger.info(f"Created models directory: {self.models_path}")
            except Exception as e:
                logger.error(f"Cannot create models directory: {e}")
                # List contents for debugging
                self._list_volume_contents()
                raise RuntimeError(f"Cannot access or create models directory: {self.models_path}")
        
        missing_models = []
        existing_models = []
        
        for model_key, config in self.model_configs.items():
            path = config["local_path"]
            if os.path.exists(path):
                # Check if the directory has content (not empty)
                try:
                    contents = os.listdir(path)
                    if contents:  # Directory exists and is not empty
                        existing_models.append(model_key)
                        logger.info(f"✓ Model {model_key} found at {path} with {len(contents)} files")
                    else:
                        missing_models.append(model_key)
                        logger.warning(f"✗ Model {model_key} directory exists but is empty: {path}")
                except Exception as e:
                    missing_models.append(model_key)
                    logger.warning(f"✗ Model {model_key} directory exists but cannot be read: {path}, error: {e}")
            else:
                missing_models.append(model_key)
                logger.warning(f"✗ Model {model_key} not found at {path}")
        
        if not existing_models:
            logger.error("No models found in volume!")
            # List what's actually in the models directory
            if os.path.exists(self.models_path):
                try:
                    contents = os.listdir(self.models_path)
                    logger.info(f"Contents of {self.models_path}: {contents}")
                except Exception as e:
                    logger.error(f"Cannot list contents of {self.models_path}: {e}")
            
            self._list_volume_contents()
            raise RuntimeError(f"No models found in volume at {self.models_path}. Please ensure models are uploaded to the network volume.")
        
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.info("These models will be downloaded at runtime if requested")
        
        # Update model configs to prioritize existing models
        self.model_configs = {k: v for k, v in self.model_configs.items() if k in existing_models}
        
        logger.info(f"Available pre-downloaded models: {list(self.model_configs.keys())}")
    
    def check_gpu_compatibility(self, model_key: str) -> bool:
        """Check if current GPU has enough memory for the specified model"""
        if not torch.cuda.is_available():
            return True
            
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        required_memory_gb = self.model_configs[model_key]["min_gpu_memory_gb"]
        
        compatible = gpu_memory_gb >= required_memory_gb
        logger.info(f"GPU Memory: {gpu_memory_gb:.1f}GB, Required: {required_memory_gb}GB, Compatible: {compatible}")
        return compatible
    
    def select_optimal_model(self, preferred_model: str = None) -> str:
        """Select the optimal model based on GPU compatibility"""
        if preferred_model and preferred_model in self.model_configs and self.check_gpu_compatibility(preferred_model):
            logger.info(f"Using preferred model: {preferred_model}")
            return preferred_model
        
        # Try models in order of preference (14B first if compatible)
        for model_key in ["wan-2.1-14b", "wan-2.1-1.3b"]:
            if model_key in self.model_configs and self.check_gpu_compatibility(model_key):
                logger.info(f"Auto-selected optimal model: {model_key}")
                return model_key
        
        # Return first available model even if not ideal
        available_models = list(self.model_configs.keys())
        if available_models:
            selected = available_models[0]
            logger.warning(f"Using first available model despite compatibility issues: {selected}")
            return selected
        
        raise RuntimeError("No compatible models available")
    
    def get_pipeline(self, model_key: str):
        """Get or create pipeline for the specified model"""
        if model_key not in self.pipelines:
            logger.info(f"Loading pipeline for model: {model_key}")
            
            config = self.model_configs[model_key]
            
            # PRIORITIZE LOCAL PATH - Use local path if available, otherwise use model_id for download
            model_path = config["local_path"] if os.path.exists(config["local_path"]) else config["model_id"]
            
            if model_path == config["local_path"]:
                logger.info(f"✓ Using pre-downloaded model from: {model_path}")
            else:
                logger.warning(f"⚠ Model not found locally, will download from HuggingFace: {config['model_id']}")
            
            try:
                # Load VAE
                logger.info("Loading VAE...")
                vae = AutoencoderKLWan.from_pretrained(
                    model_path, 
                    subfolder="vae", 
                    torch_dtype=torch.float32
                )
                
                # Setup scheduler
                logger.info("Setting up scheduler...")
                scheduler = UniPCMultistepScheduler(
                    prediction_type='flow_prediction',
                    use_flow_sigmas=True,
                    num_train_timesteps=1000,
                    flow_shift=config["flow_shift"]
                )
                
                # Load pipeline
                logger.info("Loading main pipeline...")
                pipe = WanPipeline.from_pretrained(
                    model_path,
                    vae=vae,
                    torch_dtype=torch.bfloat16
                )
                pipe.scheduler = scheduler
                pipe.to(self.device)
                
                self.pipelines[model_key] = pipe
                logger.info(f"✓ Pipeline loaded successfully for {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to load pipeline for {model_key}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        return self.pipelines[model_key]
    
    def generate_video(self, prompt: str, reference_id: str, duration: float = 3.0, 
                      resolution: str = None, model: str = None, 
                      guidance_scale: float = 5.0, num_frames: int = 81,
                      negative_prompt: str = None) -> Dict[str, Any]:
        """Generate video from text prompt using official Wan2.1 method"""
        try:
            model = model or self.select_optimal_model()
            
            if model not in self.model_configs:
                raise ValueError(f"Model {model} not available. Available models: {list(self.model_configs.keys())}")
            
            model_config = self.model_configs[model]
            
            # Set resolution
            if resolution:
                width, height = map(int, resolution.split('x'))
            else:
                width = model_config["recommended_size"]["width"]
                height = model_config["recommended_size"]["height"]
            
            # Default negative prompt if not provided
            if negative_prompt is None:
                negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            
            # Get pipeline
            pipe = self.get_pipeline(model)
            
            # Generate video
            logger.info(f"Generating video with model {model}, resolution {width}x{height}, frames {num_frames}")
            logger.info(f"Prompt: {prompt[:100]}...")
            
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
            ).frames[0]
            
            # Save video to temporary location
            output_dir = f"/tmp/outputs/{uuid.uuid4().hex}"
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, f"{reference_id}.mp4")
            
            # Export video
            logger.info(f"Exporting video to: {video_path}")
            export_to_video(output, video_path, fps=16)
            
            # Upload to Supabase storage
            logger.info("Uploading video to storage...")
            video_url = self.upload_to_storage(video_path, reference_id)
            
            # Clean up
            shutil.rmtree(output_dir, ignore_errors=True)
            
            return {
                "video_url": video_url,
                "model_used": model,
                "status": "success",
                "reference_id": reference_id,
                "resolution": f"{width}x{height}",
                "num_frames": num_frames,
                "used_local_model": os.path.exists(model_config["local_path"])  # Added this field
            }
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def handler(job):
    """RunPod handler function"""
    try:
        job_input = job.get("input", {})
        
        if job_input.get("action") == "get_model_info":
            return {
                "available_models": list(video_generator.model_configs.keys()),
                "device": video_generator.device,
                "models_path": video_generator.models_path,
                "model_details": {
                    k: {
                        "local_available": os.path.exists(v["local_path"]),
                        "local_path": v["local_path"],
                        "recommended_size": v["recommended_size"],
                        "min_gpu_memory_gb": v["min_gpu_memory_gb"]
                    } for k, v in video_generator.model_configs.items()
                }
            }
        
        # Debug action to check volume
        if job_input.get("action") == "debug_volume":
            video_generator._list_volume_contents()
            return {
                "models_path": video_generator.models_path,
                "available_models": list(video_generator.model_configs.keys()),
                "model_paths_exist": {
                    model: os.path.exists(config["local_path"]) 
                    for model, config in video_generator.model_configs.items()
                }
            }
        
        # Validate required inputs
        prompt = job_input.get("prompt")
        reference_id = job_input.get("reference_id")
        
        if not prompt or not reference_id:
            return {"error": "Both prompt and reference_id are required"}
        
        # Generate video
        result = video_generator.generate_video(
            prompt=prompt,
            reference_id=reference_id,
            duration=job_input.get("duration", 3.0),
            resolution=job_input.get("resolution"),
            model=job_input.get("model"),
            guidance_scale=job_input.get("guidance_scale", 5.0),
            num_frames=job_input.get("num_frames", 81),
            negative_prompt=job_input.get("negative_prompt")
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

# Initialize generator
try:
    video_generator = WanVideoGenerator()
    logger.info("Video generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize video generator: {e}")
    raise

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Missing Supabase credentials")
        sys.exit(1)
    
    logger.info("Starting RunPod worker...")
    runpod.serverless.start({"handler": handler})