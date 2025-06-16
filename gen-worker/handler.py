import runpod
import torch
import os
import tempfile
import base64
from diffusers import DiffusionPipeline
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

# Configure logging
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

class VideoGenerator:
    def __init__(self):
        self.pipeline = None
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Enable cuDNN autotuner
        if self.device == "cuda":
            cudnn.benchmark = True
            cudnn.deterministic = False
            
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Update model configurations with optimized settings
        self.model_configs = {
            "wan-2.1-14b": {
                "name": "alibaba-pai/Wan-2.1-T2V-14B",
                "memory_optimization": True,
                "recommended_steps": 50,
                "attention_slicing": True,
                "channels_last": True,
                "batch_size": 1
            },
            "wan-2.1-1.3b": {
                "name": "alibaba-pai/Wan-2.1-T2V-1.3B",
                "memory_optimization": False,
                "recommended_steps": 30,
                "attention_slicing": False,
                "channels_last": True,
                "batch_size": 1
            }
        }
        
    def load_model(self, model_key: str = "wan-2.1-14b"):
        """Load the specified Wan model with optimizations"""
        try:
            if self.pipeline is not None and self.current_model == model_key:
                return True
                
            with torch_gc():
                if self.pipeline is not None:
                    del self.pipeline
                
                model_config = self.model_configs.get(model_key)
                if not model_config:
                    raise ValueError(f"Unknown model: {model_key}")
                
                logger.info(f"Loading {model_key} model...")
                
                # Optimize memory usage during model loading
                torch.cuda.empty_cache()
                
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_config["name"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                )
                
                if self.device == "cuda":
                    self.pipeline = self.pipeline.to(self.device)
                    
                    # Apply memory and performance optimizations
                    if model_config["memory_optimization"]:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        self.pipeline.enable_model_cpu_offload()
                    
                    if model_config["attention_slicing"]:
                        self.pipeline.enable_attention_slicing(1)
                    
                    if model_config["channels_last"]:
                        self.pipeline.unet = self.pipeline.unet.to(memory_format=torch.channels_last)
                
                self.current_model = model_key
                logger.info(f"Model {model_key} loaded with optimizations")
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_video(self, prompt: str, duration: float = 3.0, resolution: str = "512x512", 
                      model: str = "wan-2.1-14b", num_inference_steps: int = None, 
                      guidance_scale: float = 7.5, return_method: str = "base64") -> Dict[str, Any]:
        """Generate video from text prompt with flexible return options"""
        with torch_gc():
            try:
                # Load model if needed
                if self.pipeline is None or self.current_model != model:
                    if not self.load_model(model):
                        raise Exception(f"Failed to load model: {model}")
                
                # Get model config for default inference steps
                model_config = self.model_configs.get(model, {})
                if num_inference_steps is None:
                    num_inference_steps = model_config.get("recommended_steps", 50)
                
                # Parse resolution
                width, height = map(int, resolution.split('x'))
                
                # Calculate number of frames based on duration (assuming 24 fps)
                fps = 24
                num_frames = int(duration * fps)
                
                logger.info(f"Generating video with {model}: {prompt}")
                logger.info(f"Resolution: {width}x{height}, Duration: {duration}s, Frames: {num_frames}")
                
                # Generate video
                with torch.autocast(self.device):
                    result = self.pipeline(
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
                
                # Save video to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Convert frames to video
                if hasattr(result, 'frames') and result.frames:
                    frames = result.frames[0]
                    self._save_frames_as_video(frames, temp_path, fps)
                elif hasattr(result, 'videos') and result.videos is not None:
                    video_tensor = result.videos[0]
                    self._save_tensor_as_video(video_tensor, temp_path, fps)
                else:
                    raise Exception("Unexpected output format from pipeline")
                
                # Return based on method
                if return_method == "base64":
                    with open(temp_path, 'rb') as f:
                        video_bytes = f.read()
                    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                    os.unlink(temp_path)
                    
                    return {
                        "video": video_base64,
                        "size_bytes": len(video_bytes),
                        "format": "base64"
                    }
                    
                elif return_method == "file_info":
                    # Return file info for external upload
                    file_size = os.path.getsize(temp_path)
                    
                    # Read file for upload
                    with open(temp_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    return {
                        "video_data": video_bytes,
                        "temp_path": temp_path,
                        "size_bytes": file_size,
                        "format": "binary"
                    }
                    
                else:
                    raise ValueError(f"Unknown return_method: {return_method}")
                    
            except Exception as e:
                logger.error(f"Error generating video: {str(e)}")
                logger.error(traceback.format_exc())
                raise e
    
    def _save_frames_as_video(self, frames, output_path: str, fps: int):
        """Save list of PIL images as video"""
        try:
            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                if hasattr(frame, 'convert'):  # PIL Image
                    frame_array = np.array(frame.convert('RGB'))
                else:  # Already numpy array
                    frame_array = frame
                frame_arrays.append(frame_array)
            
            # Save as video using imageio
            imageio.mimsave(output_path, frame_arrays, fps=fps, codec='libx264')
            
        except Exception as e:
            logger.error(f"Error saving frames as video: {str(e)}")
            raise e
    
    def _save_tensor_as_video(self, video_tensor, output_path: str, fps: int):
        """Save video tensor as video file"""
        try:
            # Convert tensor to numpy array
            if torch.is_tensor(video_tensor):
                video_array = video_tensor.cpu().numpy()
            else:
                video_array = video_tensor
            
            # Ensure correct shape and data type
            if video_array.dtype != np.uint8:
                video_array = (video_array * 255).astype(np.uint8)
            
            # Handle different tensor formats
            if len(video_array.shape) == 4:  # (frames, height, width, channels)
                frames = video_array
            elif len(video_array.shape) == 5:  # (batch, frames, channels, height, width)
                frames = video_array[0].transpose(0, 2, 3, 1)  # Convert to (frames, height, width, channels)
            else:
                raise ValueError(f"Unexpected video tensor shape: {video_array.shape}")
            
            # Save as video
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
            
        except Exception as e:
            logger.error(f"Error saving tensor as video: {str(e)}")
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
video_generator = VideoGenerator()
supabase_uploader = SupabaseUploader()

def handler(job):
    """RunPod handler function"""
    try:
        # Get job input
        job_input = job.get("input", {})
        
        # Extract parameters
        prompt = job_input.get("prompt", "")
        duration = job_input.get("duration", 3.0)
        resolution = job_input.get("resolution", "512x512")
        model = job_input.get("model", "wan-2.1-14b")
        num_inference_steps = job_input.get("num_inference_steps")
        guidance_scale = job_input.get("guidance_scale", 7.5)
        
        # Output options
        output_method = job_input.get("output_method", "base64")  # base64, supabase, or both
        supabase_bucket = job_input.get("supabase_bucket", "videos")
        custom_filename = job_input.get("filename")
        
        # Validate inputs
        if not prompt:
            return {"error": "Prompt is required"}
        
        # Validate model
        if model not in video_generator.model_configs:
            available_models = list(video_generator.model_configs.keys())
            return {"error": f"Invalid model: {model}. Available: {available_models}"}
        
        # Validate resolution format
        try:
            width, height = map(int, resolution.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError()
        except:
            return {"error": "Invalid resolution format. Use format like '512x512'"}
        
        # Validate duration
        if duration <= 0 or duration > 30:
            return {"error": "Duration must be between 0 and 30 seconds"}
        
        # Validate output method
        valid_methods = ["base64", "supabase", "both"]
        if output_method not in valid_methods:
            return {"error": f"Invalid output_method: {output_method}. Valid: {valid_methods}"}
        
        logger.info(f"Processing job - Model: {model}, Prompt: {prompt[:50]}...")
        
        # Generate video
        if output_method in ["supabase", "both"]:
            # Generate with file info for upload
            result = video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                model=model,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                return_method="file_info"
            )
            
            response = {
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "model": model,
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
            
            # Clean up temp file
            if os.path.exists(result["temp_path"]):
                os.unlink(result["temp_path"])
                
        else:
            # Generate with base64 only
            result = video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                model=model,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                return_method="base64"
            )
            
            response = {
                "video": result["video"],
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "model": model,
                "size_bytes": result["size_bytes"],
                "status": "success"
            }
        
        return response
        
    except Exception as e:
        error_msg = f"Error processing job: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    
    # Pre-load default model if needed
    try:
        video_generator.load_model("wan-2.1-14b")
        logger.info("Default model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {str(e)}")
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})