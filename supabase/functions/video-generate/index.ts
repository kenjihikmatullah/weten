import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { corsHeaders } from '../_shared/cors.ts'
import { GenerateVideoRequest, GenerateVideoResponse } from '../_shared/video/types.ts'
import { DbClient } from '../_shared/video/db.ts'
import { RunpodClient } from '../_shared/video/runpod.ts'
import { validateAuthToken } from "../_shared/auth.ts"

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Not found' }), { 
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 404 
    })
  }

  try {
    // Validate request body
    const body = await req.json() as GenerateVideoRequest
    if (!body.prompt) {
      return new Response(
        JSON.stringify({ error: 'Prompt is required' }), 
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const { prompt, resolution = "854x480", duration = 3.0, form = 'short' } = body
    const user = await validateAuthToken(req)

    console.log('Starting video generation:', prompt.substring(0, 50))

    // Create DB record
    const dbClient = new DbClient()
    const videoGeneration = await dbClient.createVideoGeneration({
      userId: user.id,
      userPrompt: prompt,
      resolution,
      duration,
      form,
      status: 'TO_PROCESS'
    })

    // Generate video via Runpod
    const runpodClient = new RunpodClient()
    const runpodId = await runpodClient.generateVideo({
      prompt,
      resolution,
      duration
    })

    // Update record with external reference
    await dbClient.updateVideoGenerationStatus(videoGeneration.id!, 'PROCESSING', runpodId)

    const response: GenerateVideoResponse = {
      success: true,
      message: 'Video generation started',
      videoGenerationId: videoGeneration.id?.toString()
    }

    return new Response(
      JSON.stringify(response),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 202 
      }
    )

  } catch (error) {
    console.error('Error:', error)
    const response: GenerateVideoResponse = {
      success: false,
      message: error.message
    }

    return new Response(
      JSON.stringify(response),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: error.message === 'Unauthorized' ? 401 : 400
      }
    )
  }
})