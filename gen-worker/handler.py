import runpod
import torch
import os
import base64
import logging
import traceback
from typing import Dict, Any
import uuid
import shutil
import subprocess
import sys
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class WanVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wan_repo_path = "/app/Wan2.1"
        
        # Detect network volume path
        self.models_path = self._detect_volume_path()
        logger.info(f"Using models path: {self.models_path}")
        
        self.model_configs = {
            "wan-2.1-1.3b": {
                "local_path": f"{self.models_path}/Wan2.1-T2V-1.3B",
                "task": "t2v-1.3B",
                "recommended_size": "480x854",
                "sample_shift": 8,
                "sample_guide_scale": 6,
                "min_gpu_memory_gb": 8
            },
            "wan-2.1-14b": {
                "local_path": f"{self.models_path}/Wan2.1-T2V-14B",
                "task": "t2v-14B",
                "recommended_size": "720x1280",
                "sample_shift": 8,
                "sample_guide_scale": 7.5,
                "min_gpu_memory_gb": 24
            }
        }
        
        self._verify_models()
        
        if self.wan_repo_path not in sys.path:
            sys.path.insert(0, self.wan_repo_path)
    
    def _detect_volume_path(self) -> str:
        """
        Detect the network volume mount path in RunPod Serverless
        """
        # Common RunPod network volume mount paths
        possible_paths = [
            "/runpod-volume",           # Most common
            "/workspace",               # Alternative
            "/volume",                  # Alternative
            "/mnt/runpod-volume",       # Alternative
            os.environ.get("RUNPOD_VOLUME_PATH", ""),  # Environment variable
        ]
        
        # Check environment variable first
        if "RUNPOD_VOLUME_PATH" in os.environ:
            env_path = os.environ["RUNPOD_VOLUME_PATH"]
            if os.path.exists(env_path):
                return f"{env_path}/models"
        
        # Check common paths
        for path in possible_paths:
            if path and os.path.exists(path):
                models_path = f"{path}/models"
                if os.path.exists(models_path):
                    return models_path
                # If models folder doesn't exist, try to create it
                try:
                    os.makedirs(models_path, exist_ok=True)
                    return models_path
                except:
                    continue
        
        # Fallback: check root filesystem for any mounted volumes
        volume_path = self._find_volume_by_content()
        if volume_path:
            return f"{volume_path}/models"
        
        # Last resort - create in container filesystem (not persistent)
        logger.warning("Network volume not found, using container filesystem (not persistent!)")
        return "/tmp/models"
    
    def _find_volume_by_content(self) -> str:
        """
        Find volume by looking for model directories
        """
        # Check common mount points
        mount_points = ["/mnt", "/media", "/volumes"]
        
        for mount_point in mount_points:
            if os.path.exists(mount_point):
                for item in os.listdir(mount_point):
                    item_path = os.path.join(mount_point, item)
                    if os.path.isdir(item_path):
                        # Check if this directory contains model folders
                        models_path = os.path.join(item_path, "models")
                        if os.path.exists(models_path):
                            for model_folder in ["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"]:
                                if os.path.exists(os.path.join(models_path, model_folder)):
                                    return item_path
        
        # Check if there are any directories that look like model directories
        for root, dirs, files in os.walk("/"):
            # Skip system directories
            if any(sys_dir in root for sys_dir in ["/proc", "/sys", "/dev", "/boot", "/etc"]):
                continue
            
            if "Wan2.1-T2V-1.3B" in dirs or "Wan2.1-T2V-14B" in dirs:
                return os.path.dirname(root) if root.endswith("/models") else root
        
        return None
    
    def _list_volume_contents(self):
        """
        Debug function to list volume contents
        """
        logger.info("=== Volume Detection Debug ===")
        
        # Check environment variables
        for key, value in os.environ.items():
            if "volume" in key.lower() or "runpod" in key.lower():
                logger.info(f"ENV: {key} = {value}")
        
        # Check common paths
        paths_to_check = ["/", "/runpod-volume", "/workspace", "/volume", "/mnt"]
        
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
            file_path = f"public/{reference_id}.mp4"
            response = supabase.storage.from_('videos').upload(
                file_path,
                file_bytes,
                {"content-type": "video/mp4"}
            )
            
            # Get public URL
            public_url = supabase.storage.from_('videos').get_public_url(file_path)
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
            if os.path.exists(path) and os.listdir(path):
                existing_models.append(model_key)
                logger.info(f"✓ Model {model_key} found at {path}")
            else:
                missing_models.append(model_key)
                logger.warning(f"✗ Model {model_key} not found at {path}")
        
        if not existing_models:
            logger.error("No models found in volume!")
            # List what's actually in the models directory
            if os.path.exists(self.models_path):
                contents = os.listdir(self.models_path)
                logger.info(f"Contents of {self.models_path}: {contents}")
            
            self._list_volume_contents()
            raise RuntimeError("No models found in volume. Please ensure models are uploaded to the network volume.")
        
        # Update model configs to only include existing models
        self.model_configs = {k: v for k, v in self.model_configs.items() if k in existing_models}
        
        logger.info(f"Available models: {list(self.model_configs.keys())}")
    
    def check_gpu_compatibility(self, model_key: str) -> bool:
        """Check if current GPU has enough memory for the specified model"""
        if not torch.cuda.is_available():
            return True
            
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        required_memory_gb = self.model_configs[model_key]["min_gpu_memory_gb"]
        
        return gpu_memory_gb >= required_memory_gb
    
    def select_optimal_model(self, preferred_model: str = None) -> str:
        """Select the optimal model based on GPU compatibility"""
        if preferred_model and preferred_model in self.model_configs and self.check_gpu_compatibility(preferred_model):
            return preferred_model
        
        for model_key in ["wan-2.1-14b", "wan-2.1-1.3b"]:
            if model_key in self.model_configs and self.check_gpu_compatibility(model_key):
                return model_key
        
        # Return first available model
        available_models = list(self.model_configs.keys())
        if available_models:
            return available_models[0]
        
        raise RuntimeError("No compatible models available")
    
    def generate_video(self, prompt: str, reference_id: str, duration: float = 3.0, 
                resolution: str = None, model: str = None, 
                guidance_scale: float = None) -> Dict[str, Any]:
        """Generate video from text prompt"""
        try:
            model = model or self.select_optimal_model()
            
            if model not in self.model_configs:
                raise ValueError(f"Model {model} not available. Available models: {list(self.model_configs.keys())}")
            
            model_config = self.model_configs[model]
            size_str = resolution or model_config["recommended_size"]
            guidance_scale = guidance_scale or model_config["sample_guide_scale"]
            
            # Import required modules from Wan2.1
            sys.path.insert(0, self.wan_repo_path)
            from modelscope.pipelines import pipeline
            from modelscope.outputs import OutputKeys
            
            # Create video generation pipeline
            pipe = pipeline('text-to-video-synthesis', 
                        model=model_config["local_path"],
                        device=self.device)
            
            # Generate video
            width, height = map(int, size_str.split('x'))
            output = pipe({
                'text': prompt,
                'size': (height, width),  # Height comes first in modelscope
                'guidance_scale': guidance_scale,
                'sample_shift': model_config["sample_shift"]
            })
            
            # Save video to temporary location
            output_dir = f"/app/outputs/{uuid.uuid4().hex}"
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, f"{reference_id}.mp4")
            
            # Get video data and save
            video_data = output[OutputKeys.OUTPUT_VIDEO]
            video_data.save(video_path)
            
            # Upload to Supabase storage
            video_url = self.upload_to_storage(video_path, reference_id)
            
            # Clean up
            shutil.rmtree(output_dir, ignore_errors=True)
            
            return {
                "video_url": video_url,
                "model_used": model,
                "status": "success",
                "reference_id": reference_id
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
                "models_path": video_generator.models_path
            }
        
        # Debug action to check volume
        if job_input.get("action") == "debug_volume":
            video_generator._list_volume_contents()
            return {
                "models_path": video_generator.models_path,
                "available_models": list(video_generator.model_configs.keys())
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
            guidance_scale=job_input.get("guidance_scale")
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": traceback.format_exc()}

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