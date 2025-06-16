# RunPod Serverless Text-to-Video Worker

High-performance text-to-video generation worker using Wan 2.1 models on RunPod serverless with Supabase integration.

## Key Features

- Generate videos from text prompts using Wan 2.1 models
- Smart GPU memory optimization
- Multiple output methods (Base64/Supabase)
- CUDA-flexible architecture
- Optimized for minimal latency

## Models

| Model | Quality | Speed | VRAM Usage |
|-------|---------|-------|------------|
| `wan-2.1-14b` | High | Slower | ~14GB |
| `wan-2.1-1.3b` | Good | Faster | ~8GB |

## Quick Start

### Deploy to RunPod

```bash
# Build image
docker build -t your-username/text-to-video-worker .

# Push to registry
docker push your-username/text-to-video-worker
```

Then deploy on RunPod Serverless with minimum 16GB VRAM GPU.

### Environment Variables

```bash
SUPABASE_URL=your-project-url
SUPABASE_ANON_KEY=your-anon-key
```

## Usage Examples

### cURL Request
```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over ocean waves",
      "duration": 4.0,
      "resolution": "512x512",
      "model": "wan-2.1-14b",
      "output_method": "supabase"
    }
  }'
```

### Python Client
```python
import runpod

runpod.api_key = "your-api-key"
response = runpod.run(
    endpoint_id="endpoint-id",
    input={
        "prompt": "A beautiful sunset over ocean waves",
        "duration": 4.0,
        "resolution": "512x512",
        "model": "wan-2.1-14b",
        "output_method": "supabase"
    }
)
```

### Edge Function
```typescript
const response = await fetch('/functions/v1/generate-video', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "A beautiful sunset over ocean waves",
    duration: 4.0,
    resolution: "512x512",
    model: "wan-2.1-14b"
  })
})
```

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Generation prompt |
| `duration` | float | No | 3.0 | Video duration (max 30s) |
| `resolution` | string | No | "512x512" | Format: "WIDTHxHEIGHT" |
| `model` | string | No | "wan-2.1-14b" | Model selection |
| `output_method` | string | No | "base64" | "base64"/"supabase"/"both" |

## Output Examples

### Supabase Method (Recommended)
```json
{
  "success": true,
  "video_url": "https://supabase-url/videos/video_123.mp4",
  "size_bytes": 2458624,
  "generation_time": 45,
  "metadata": {
    "prompt": "input-prompt",
    "model": "wan-2.1-14b",
    "resolution": "512x512"
  }
}
```

### Base64 Method
```json
{
  "video": "base64-encoded-data",
  "size_bytes": 2458624,
  "status": "success"
}
```

## Performance Tips

- Use `wan-2.1-1.3b` for testing/development
- Prefer Supabase output for videos >10MB
- Keep resolution ≤768x768 for faster generation
- Use shorter durations when possible

## Error Handling

Common error responses:
```json
{
  "error": "Invalid model specified",
  "available_models": ["wan-2.1-14b", "wan-2.1-1.3b"]
}
```
```json
{
  "error": "GPU memory insufficient",
  "minimum_required": "16GB VRAM"
}
```

## Requirements

- GPU: 16GB+ VRAM (A4000/A5000/A100)
- CUDA compatible
- FFmpeg installed
- Python 3.8+

For detailed technical docs and advanced configurations, visit our [Wiki](https://github.com/your-repo/wiki).