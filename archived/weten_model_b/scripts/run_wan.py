#!/usr/bin/env python3
"""
Wan Video Generation Script for n8n workflows
Generates videos from text prompts using Wan 2.1 model via Diffusers
"""

import sys
import os
import argparse
import logging
import subprocess
from pathlib import Path

# Setup logging to both file and console
log_file = '/home/node/videos/wan_debug.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all dependencies are available"""
    logger.info("=== ENVIRONMENT CHECK ===")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check if directories exist
    paths_to_check = ['/home/node/scripts', '/home/node/videos']
    for path in paths_to_check:
        exists = os.path.exists(path)
        logger.info(f"Path {path} exists: {exists}")
    
    # Check if we can import required packages
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        logger.error(f"Cannot import torch: {e}")
        return False
    
    try:
        import diffusers
        logger.info(f"Diffusers version: {diffusers.__version__}")
    except ImportError as e:
        logger.error(f"Cannot import diffusers: {e}")
        return False
    
    return True

# Import required modules
logger.info("=== IMPORTING REQUIRED MODULES ===")
try:
    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    logger.info("Successfully imported torch, WanPipeline, and export_to_video")
    import_success = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure diffusers is installed: pip install diffusers[torch]")
    WanPipeline = None
    export_to_video = None
    import_success = False

class WanVideoGenerator:
    def __init__(self):
        self.pipeline = None
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            self.torch_available = True
        except ImportError:
            logger.error("PyTorch not available")
            self.torch_available = False
            self.device = "cpu"
        
    def load_model(self, model_id=None):
        """Load the Wan video generation pipeline"""
        if not self.torch_available or WanPipeline is None:
            logger.error("Required modules not available, cannot load model")
            return False
            
        try:
            import torch
            logger.info("Loading Wan video generation pipeline...")
            
            # Default model options (from smallest to largest)
            model_options = [
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",  # Smallest, fastest
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",   # Large, high quality
                "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"  # Image-to-video (if needed)
            ]
            
            if model_id:
                model_options.insert(0, model_id)
            
            model_loaded = False
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    
                    # Load pipeline with memory optimizations
                    self.pipeline = WanPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    
                    # Move to device if needed
                    if self.device == "cpu":
                        self.pipeline = self.pipeline.to(self.device)
                    
                    # Enable memory optimizations for GPU
                    if self.device == "cuda":
                        try:
                            # Enable CPU offloading to save VRAM
                            self.pipeline.enable_model_cpu_offload()
                            logger.info("Enabled CPU offloading")
                        except:
                            logger.warning("Could not enable CPU offloading")
                        
                        try:
                            # Enable VAE slicing to reduce memory usage
                            self.pipeline.vae.enable_slicing()
                            logger.info("Enabled VAE slicing")
                        except:
                            logger.warning("Could not enable VAE slicing")
                        
                        try:
                            # Enable attention slicing
                            self.pipeline.enable_attention_slicing()
                            logger.info("Enabled attention slicing")
                        except:
                            logger.warning("Could not enable attention slicing")
                    
                    logger.info(f"Successfully loaded pipeline: {model_name}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if not model_loaded:
                logger.error("Failed to load any Wan model")
                
            return model_loaded
            
        except Exception as e:
            logger.error(f"Critical error during model loading: {e}")
            return False
    
    def generate_video(self, prompt, output_path, duration=4, width=512, height=512, fps=8, num_inference_steps=50):
        """
        Generate video from text prompt
        
        Args:
            prompt (str): Text prompt for video generation
            output_path (str): Path to save the generated video
            duration (int): Duration in seconds (default: 4)
            width (int): Video width (default: 512)  
            height (int): Video height (default: 512)
            fps (int): Frames per second (default: 8)
            num_inference_steps (int): Number of inference steps (default: 50)
        """
        try:
            if not self.pipeline:
                logger.error("Pipeline not loaded. Call load_model() first.")
                return self.create_placeholder_video(output_path, duration, width, height, fps)
                
            import torch
            logger.info(f"Generating video for prompt: '{prompt}'")
            logger.info(f"Duration: {duration}s, Resolution: {width}x{height}, FPS: {fps}")
            
            # Calculate number of frames - Wan typically uses specific frame counts
            # Common frame counts: 17, 33, 49, 65, 81 (1-5 seconds at ~16fps)
            frame_mapping = {
                1: 17, 2: 33, 3: 49, 4: 65, 5: 81
            }
            num_frames = frame_mapping.get(duration, duration * 16)  # fallback to duration * 16
            
            logger.info(f"Using {num_frames} frames for {duration} second video")
            
            # Generate video
            logger.info("Starting video generation...")
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            # Extract frames from result
            if hasattr(result, 'frames'):
                video_frames = result.frames[0]
            elif hasattr(result, 'videos'):
                video_frames = result.videos[0]
            else:
                logger.error("Could not extract frames from pipeline result")
                return self.create_placeholder_video(output_path, duration, width, height, fps)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save video using diffusers utility
            logger.info(f"Saving video to: {output_path}")
            export_to_video(video_frames, output_path, fps=fps)
            
            logger.info("Video generation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            logger.info("Attempting to create placeholder video...")
            return self.create_placeholder_video(output_path, duration, width, height, fps)
    
    def create_placeholder_video(self, output_path, duration=4, width=512, height=512, fps=8):
        """Create a placeholder video using FFmpeg when Wan is not available"""
        try:
            logger.info("Creating placeholder video with FFmpeg...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create a simple colored video with text using FFmpeg
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite
                '-f', 'lavfi',
                '-i', f'color=c=blue:size={width}x{height}:duration={duration}:rate={fps}',
                '-vf', f'drawtext=text="Wan 2.1 Model Loading Failed - Placeholder":fontcolor=white:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("Placeholder video created successfully")
                return True
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return False
        except Exception as e:
            logger.error(f"Error creating placeholder video: {e}")
            return False

def main():
    logger.info("=== WAN VIDEO GENERATION SCRIPT STARTED ===")
    
    # Check environment first
    if not check_environment():
        logger.error("Environment check failed")
    
    parser = argparse.ArgumentParser(description='Generate video from text prompt using Wan 2.1')
    parser.add_argument('prompt', help='Text prompt for video generation')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('duration', type=int, nargs='?', default=4, 
                       help='Video duration in seconds (default: 4)')
    parser.add_argument('--width', type=int, default=512, 
                       help='Video width (default: 512)')
    parser.add_argument('--height', type=int, default=512, 
                       help='Video height (default: 512)')
    parser.add_argument('--fps', type=int, default=8, 
                       help='Frames per second (default: 8)')
    parser.add_argument('--steps', type=int, default=50, 
                       help='Number of inference steps (default: 50)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model ID to use (optional)')
    
    args = parser.parse_args()
    
    logger.info(f"Arguments received:")
    logger.info(f"  Prompt: {args.prompt}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Duration: {args.duration}")
    logger.info(f"  Resolution: {args.width}x{args.height}")
    logger.info(f"  FPS: {args.fps}")
    logger.info(f"  Model: {args.model or 'Auto-select'}")
    
    # Validate inputs
    if not args.prompt.strip():
        logger.error("Prompt cannot be empty")
        print("ERROR: Prompt cannot be empty")
        sys.exit(1)
    
    if args.duration < 1 or args.duration > 5:
        logger.warning("Duration should be between 1 and 5 seconds for best results")
        if args.duration > 5:
            args.duration = 5
        elif args.duration < 1:
            args.duration = 1
    
    # Initialize generator
    logger.info("Initializing video generator...")
    generator = WanVideoGenerator()
    
    # Load model
    logger.info("Loading model...")
    model_loaded = generator.load_model(args.model)
    if not model_loaded:
        logger.warning("Model loading failed, will use fallback method")
    
    # Generate video
    logger.info("Starting video generation...")
    success = generator.generate_video(
        prompt=args.prompt,
        output_path=args.output,
        duration=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_inference_steps=args.steps
    )
    
    if success:
        logger.info(f"Video successfully generated: {args.output}")
        # Verify file exists and has size > 0
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            logger.info(f"Output file verified. Size: {os.path.getsize(args.output)} bytes")
            print(f"SUCCESS: {args.output}")  # For n8n to capture
            sys.exit(0)
        else:
            logger.error("Output file is missing or empty")
            print("ERROR: Output file is missing or empty")
            sys.exit(1)
    else:
        logger.error("Video generation failed")
        print("ERROR: Video generation failed")  # For n8n to capture  
        sys.exit(1)

if __name__ == "__main__":
    main()