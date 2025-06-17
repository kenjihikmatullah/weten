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
        self.models_path = "/workspace/models"
        
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
    
    def upload_to_storage(self, file_path: str, generation_id: str) -> str:
        """Upload video to Supabase storage and return public URL"""
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Upload to storage bucket 'videos'
            file_path = f"public/{generation_id}.mp4"
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
        for model_key, config in self.model_configs.items():
            path = config["local_path"]
            if not os.path.exists(path) or not os.listdir(path):
                raise RuntimeError(f"Model {model_key} not found in volume at {path}")
        logger.info("All models verified in volume")
    
    def check_gpu_compatibility(self, model_key: str) -> bool:
        """Check if current GPU has enough memory for the specified model"""
        if not torch.cuda.is_available():
            return True
            
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        required_memory_gb = self.model_configs[model_key]["min_gpu_memory_gb"]
        
        return gpu_memory_gb >= required_memory_gb
    
    def select_optimal_model(self, preferred_model: str = None) -> str:
        """Select the optimal model based on GPU compatibility"""
        if preferred_model and self.check_gpu_compatibility(preferred_model):
            return preferred_model
        
        for model_key in ["wan-2.1-14b", "wan-2.1-1.3b"]:
            if self.check_gpu_compatibility(model_key):
                return model_key
        
        return "wan-2.1-1.3b"
    
    def generate_video(self, prompt: str, generation_id: str, duration: float = 3.0, 
                      resolution: str = None, model: str = None, 
                      guidance_scale: float = None) -> Dict[str, Any]:
        """Generate video from text prompt"""
        try:
            model = model or self.select_optimal_model()
            model_config = self.model_configs[model]
            
            size_str = resolution or model_config["recommended_size"]
            guidance_scale = guidance_scale or model_config["sample_guide_scale"]
            
            output_dir = f"/app/outputs/{uuid.uuid4().hex}"
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                "python", f"{self.wan_repo_path}/generate.py",
                "--task", model_config["task"],
                "--size", size_str,
                "--ckpt_dir", model_config["local_path"],
                "--sample_shift", str(model_config["sample_shift"]),
                "--sample_guide_scale", str(guidance_scale),
                "--prompt", prompt,
                "--output_dir", output_dir
            ]
            
            os.chdir(self.wan_repo_path)
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            video_file = next(
                (os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                 if f.endswith(('.mp4', '.avi', '.mov'))), 
                None
            )
            
            if not video_file:
                raise Exception("Video generation failed")
            
            # Upload to Supabase storage
            video_url = self.upload_to_storage(video_file, generation_id)
            
            # Clean up
            shutil.rmtree(output_dir, ignore_errors=True)
            
            return {
                "video_url": video_url,
                "model_used": model,
                "status": "success",
                "generation_id": generation_id
            }
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job.get("input", {})
        
        if job_input.get("action") == "get_model_info":
            return {
                "available_models": list(video_generator.model_configs.keys()),
                "device": video_generator.device
            }
        
        # Validate required inputs
        prompt = job_input.get("prompt")
        generation_id = job_input.get("generation_id")
        
        if not prompt or not generation_id:
            return {"error": "Both prompt and generation_id are required"}
        
        # Generate video
        result = video_generator.generate_video(
            prompt=prompt,
            generation_id=generation_id,
            duration=job_input.get("duration", 3.0),
            resolution=job_input.get("resolution"),
            model=job_input.get("model"),
            guidance_scale=job_input.get("guidance_scale")
        )
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

# Initialize generator
video_generator = WanVideoGenerator()

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Missing Supabase credentials")
        sys.exit(1)
    
    logger.info("Starting RunPod worker...")
    runpod.serverless.start({"handler": handler})