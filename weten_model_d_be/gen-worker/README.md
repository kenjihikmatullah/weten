# RunPod Serverless Text-to-Video Worker

Worker untuk generate video dari text prompt menggunakan Wan 2.1-T2V-14B model di RunPod serverless.

## Features

- Text-to-video generation menggunakan Wan 2.1-T2V-14B
- Parameterized input: prompt, duration, resolution
- GPU optimization dengan xFormers dan memory management
- Base64 output untuk easy integration
- Error handling dan logging

## Setup

### 1. Build Docker Image

```bash
docker build -t your-username/text-to-video-worker .
```

### 2. Push ke Docker Registry

```bash
docker push your-username/text-to-video-worker
```

### 3. Deploy ke RunPod

1. Login ke RunPod dashboard
2. Go to Serverless → Templates
3. Create new template:
   - Container Image: `your-username/text-to-video-worker`
   - Container Registry Credentials: (jika private registry)
   - GPU: pilih yang sesuai (minimal 16GB VRAM recommended)

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text prompt untuk generate video |
| `duration` | float | No | 3.0 | Durasi video dalam detik (max 30s) |
| `resolution` | string | No | "512x512" | Resolusi video (format: "WIDTHxHEIGHT") |
| `num_inference_steps` | int | No | 50 | Jumlah inference steps |
| `guidance_scale` | float | No | 7.5 | Guidance scale untuk generation |

## Example Usage

### RunPod API Call

```python
import runpod
import base64

# Initialize RunPod
runpod.api_key = "your-api-key"

# Create job
job = runpod.run(
    endpoint_id="your-endpoint-id",
    job_input={
        "prompt": "A beautiful sunset over ocean waves, cinematic",
        "duration": 4.0,
        "resolution": "768x512",
        "num_inference_steps": 50,
        "guidance_scale": 7.5
    }
)

# Get result
result = runpod.status(job['id'])
if result['status'] == 'COMPLETED':
    video_base64 = result['output']['video']
    
    # Save video
    with open('output.mp4', 'wb') as f:
        f.write(base64.b64decode(video_base64))
```

### cURL Example

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/run \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing in a garden, realistic style",
      "duration": 3.0,
      "resolution": "512x512"
    }
  }'
```

## Output Format

```json
{
  "video": "base64-encoded-video-data",
  "prompt": "input-prompt",
  "duration": 3.0,
  "resolution": "512x512",
  "status": "success"
}
```

## Testing Locally

```bash
# Test dengan sample input
python3 handler.py

# Atau test dengan runpod local
pip install runpod
runpod test --input test_input.json
```

## Notes

- Model akan di-download saat pertama kali dijalankan (sekitar 14GB)
- Recommended GPU: A100 atau V100 dengan minimal 16GB VRAM
- Generation time tergantung kompleksitas prompt dan duration
- Video output dalam format MP4 dengan codec H.264

## Troubleshooting

1. **Out of Memory Error**: Kurangi resolution atau gunakan GPU dengan VRAM lebih besar
2. **Model Loading Error**: Pastikan internet connection stabil dan HuggingFace access
3. **Generation Slow**: Kurangi `num_inference_steps` atau `duration`

## Future Improvements

- [ ] Support multiple aspect ratios
- [ ] Batch processing
- [ ] Custom model fine-tuning integration
- [ ] Advanced scheduling options
- [ ] Video upscaling post-processing