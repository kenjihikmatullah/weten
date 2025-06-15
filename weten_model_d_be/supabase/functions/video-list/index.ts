import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { corsHeaders } from '../_shared/cors.ts'
import { validateAuthToken } from "../_shared/auth.ts"
import { DbClient } from '../_shared/video/db.ts'

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  if (req.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), { 
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 405 
    })
  }

  try {
    const user = await validateAuthToken(req)
    const dbClient = new DbClient()
    
    // Check and update completed videos first
    await dbClient.checkAndUpdateCompletedVideos(user.id)
    
    // Then get the updated list
    const videos = await dbClient.getVideosList(user.id)

    return new Response(
      JSON.stringify(videos),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    )

  } catch (error) {
    console.error('Error:', error)
    
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: error.message === 'Unauthorized' ? 401 : 500
      }
    )
  }
})