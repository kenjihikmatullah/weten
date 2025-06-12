import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { corsHeaders } from '../_shared/cors.ts'
import { GenerateVideoRequest } from './tools/types.ts'
import { DbClient } from './tools/db.ts'
import { RunpodClient } from './tools/runpod.ts'
import { validateAuthToken } from "./tools/auth.ts"

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
    const { prompt, resolution, duration, form } = await req.json() as GenerateVideoRequest

    const user = await validateAuthToken(req)

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

    return new Response(
      JSON.stringify({ 
        message: 'Video generation started',
        id: videoGeneration.id
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 202 
      }
    )

  } catch (error) {
    console.error('Error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: error.message === 'Unauthorized' ? 401 : 400
      }
    )
  }
})