#!/usr/bin/env python3
"""
Debug script to test Wan installation and dependencies
"""

import sys
import os
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=== WAN DEBUG SCRIPT ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check current working directory
    print(f"Current directory: {os.getcwd()}")
    
    # Check if directories exist
    directories = ['/opt/Wan2.1', '/home/node/scripts', '/home/node/videos', '/home/node']
    for directory in directories:
        exists = os.path.exists(directory)
        print(f"Directory {directory} exists: {exists}")
        if exists:
            try:
                contents = os.listdir(directory)
                print(f"  Contents: {contents[:10]}")  # First 10 items
            except PermissionError:
                print(f"  Permission denied to list contents")
            except Exception as e:
                print(f"  Error listing contents: {e}")
    
    # Check Python path
    print(f"Python path: {sys.path[:5]}")  # First 5 entries
    
    # Try to import torch
    print("\n=== TORCH TEST ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"Cannot import PyTorch: {e}")
    
    # Check pip packages
    print("\n=== PIP PACKAGES ===")
    try:
        result = subprocess.run(['pip3', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"Total packages: {len(lines)-2}")  # Exclude header lines
            # Show packages that might be relevant
            relevant_packages = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['torch', 'wan', 'video', 'diffus', 'transform']):
                    relevant_packages.append(line)
            print("Relevant packages:")
            for pkg in relevant_packages:
                print(f"  {pkg}")
        else:
            print(f"pip list failed: {result.stderr}")
    except Exception as e:
        print(f"Error running pip list: {e}")
    
    # Test FFmpeg
    print("\n=== FFMPEG TEST ===")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            print(f"FFmpeg: {first_line}")
        else:
            print(f"FFmpeg not working: {result.stderr}")
    except FileNotFoundError:
        print("FFmpeg not found")
    except Exception as e:
        print(f"Error testing FFmpeg: {e}")
    
    # Test file creation in videos directory
    print("\n=== FILE WRITE TEST ===")
    test_file = '/home/node/videos/test.txt'
    try:
        os.makedirs('/home/node/videos', exist_ok=True)
        with open(test_file, 'w') as f:
            f.write('test')
        print(f"Can write to {test_file}: True")
        os.remove(test_file)
    except Exception as e:
        print(f"Cannot write to {test_file}: {e}")
    
    # Try to import Wan modules
    print("\n=== WAN IMPORT TEST ===")
    sys.path.insert(0, '/opt/Wan2.1')
    
    import_attempts = [
        ('wan_video', 'from wan_video import WanVideoPipeline'),
        ('wan', 'import wan'),
        ('src.wan_video', 'from src.wan_video import WanVideoPipeline'),
    ]
    
    for name, import_cmd in import_attempts:
        try:
            exec(import_cmd)
            print(f"✓ Successfully imported: {name}")
        except ImportError as e:
            print(f"✗ Failed to import {name}: {e}")
        except Exception as e:
            print(f"✗ Error importing {name}: {e}")
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    main()