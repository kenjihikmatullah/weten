# RunPod Serverless Text-to-Video Worker

Worker untuk generate video dari text prompt menggunakan Wan 2.1 models di RunPod serverless dengan optimized Supabase integration.

## Features

- **Multi-model support**: Wan 2.1-T2V-14B (high quality) dan Wan 2.1-T2V-1.3B (faster)
- **Flexible parameters**: prompt, duration, resolution, model selection
- **Smart memory management**: Automatic optimization based on model size
- **Multiple output methods**: Base64, Supabase upload, atau both
- **Supabase integration**: Direct upload ke Supabase Storage
- **Optimized transfer**: Efficient handling untuk Supabase Edge Functions

## Transfer Method Recommendations

### 🎯 **RECOMMENDED for Supabase Integration**
```json
{
  "output_method": "supabase",
  "supabase_bucket": "videos"
}
```
**Why?**
- ✅ No size limits (base64 has ~33% overhead)
- ✅ Direct CDN URL untuk client access
- ✅ Automatic cleanup dari RunPod worker
- ✅ Faster transfer ke Edge Functions
- ✅ Better memory efficiency

### ⚠️ **NOT Recommended for Large Videos**
```json
{
  "output_method": "base64"
}
```
**Limitations:**
- ❌ 33% size increase due to base64 encoding
- ❌ Edge Functions timeout risk (>25MB videos)
- ❌ Memory intensive untuk transfer
- ❌ Manual handling needed

## Available Models

| Model | Size | Speed | Quality | Memory | Recommended Use |
|-------|------|--------|---------|---------|-----------------|
| `wan-2.1-14b` | 14B | Slower | Highest | High | Production, high-quality |
| `wan-2.1-1.3b` | 1.3B | Faster | Good | Lower | Development, quick tests |

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
| `model` | string | No | "wan-2.1-14b" | Model selection: "wan-2.1-14b" or "wan-2.1-1.3b" |
| `num_inference_steps` | int | No | Auto | Inference steps (auto-set based on model) |
| `guidance_scale` | float | No | 7.5 | Guidance scale untuk generation |
| `output_method` | string | No | "base64" | "base64", "supabase", or "both" |
| `supabase_bucket` | string | No | "videos" | Supabase Storage bucket name |
| `filename` | string | No | Auto | Custom filename (auto-generated if not provided) |

## Environment Variables

For Supabase integration, set these in your RunPod template:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

## Example Usage

### 🚀 Recommended: Supabase Integration

```python
import runpod

runpod.api_key = "your-api-key"

# Optimal for Supabase Edge Functions
job = runpod.run(
    endpoint_id="your-endpoint-id",
    job_input={
        "prompt": "A beautiful sunset over ocean waves, cinematic",
        "duration": 4.0,
        "resolution": "768x512",
        "model": "wan-2.1-1.3b",  # Faster for development
        "output_method": "supabase",
        "supabase_bucket": "videos",
        "filename": "sunset_ocean.mp4"
    }
)

result = runpod.status(job['id'])
if result['status'] == 'COMPLETED':
    video_url = result['output']['supabase']['url']
    print(f"Video available at: {video_url}")
```

### Supabase Edge Function Call

```typescript
// From your frontend/client
const response = await fetch('/functions/v1/generate-video', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "A cat playing in a garden, realistic style",
    duration: 3.0,
    resolution: "512x512",
    model: "wan-2.1-1.3b",
    output_method: "supabase"
  })
})

const result = await response.json()
console.log('Video URL:', result.video_url)
```

### Base64 Method (untuk small videos only)

```python
# Only for short/small videos
job = runpod.run(
    endpoint_id="your-endpoint-id",
    job_input={
        "prompt": "Quick animation, 2 seconds",
        "duration": 2.0,
        "resolution": "256x256",
        "model": "wan-2.1-1.3b",
        "output_method": "base64"
    }
)
```

## Output Formats

### Supabase Method (Recommended)
```json
{
  "success": true,
  "video_url": "https://your-project.supabase.co/storage/v1/object/public/videos/video_123.mp4",
  "filename": "video_123.mp4",
  "prompt": "input-prompt",
  "duration": 3.0,
  "resolution": "512x512",
  "model": "wan-2.1-1.3b",
  "size_bytes": 2458624,
  "generation_time_seconds": 45
}
```

### Base64 Method
```json
{
  "video": "base64-encoded-video-data",
  "prompt": "input-prompt", 
  "duration": 3.0,
  "resolution": "512x512",
  "model": "wan-2.1-1.3b",
  "size_bytes": 2458624,
  "status": "success"
}
```

### Both Methods
```json
{
  "video_url": "https://storage-url.mp4",
  "video_base64": "base64-data",
  "filename": "custom_name.mp4",
  "supabase": {
    "url": "https://storage-url.mp4",
    "filename": "custom_name.mp4"
  },
  "size_bytes": 2458624,
  "status": "success"
}
```

## Performance Comparison

| Aspect | Base64 Transfer | Supabase Upload |
|--------|----------------|-----------------|
| **Size Overhead** | +33% | No overhead |
| **Transfer Speed** | Slower (large payload) | Faster (URL only) |
| **Memory Usage** | High | Low |
| **Edge Function Timeout** | Risk for >10MB | No risk |
| **Client Access** | Immediate | CDN optimized |
| **Cleanup** | Manual | Automatic |

## Deployment Architecture

```
Client → Supabase Edge Function → RunPod Worker → Supabase Storage
    ↑                                                     ↓
    └─────────────────── Video URL ←─────────────────────┘
```

**Benefits:**
- Edge Function hanya transfer URL (lightweight)
- Video tersimpan permanent di Supabase Storage
- Client dapat akses video via CDN URL
- Automatic cleanup di RunPod worker

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