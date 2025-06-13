import { createClient, SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2.39.0'
import { VideoGeneration } from './types.ts'

export class DbClient {
  private client: SupabaseClient

  constructor() {
    this.client = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )
  }

  async createVideoGeneration(data: Omit<VideoGeneration, 'id' | 'createdAt' | 'updatedAt'>): Promise<VideoGeneration> {
    const { data: result, error } = await this.client
      .from('video_generations')
      .insert(data)
      .select()
      .single()

    if (error) throw new Error(`Database error: ${error.message}`)
    return result
  }

  async updateVideoGenerationStatus(id: number, status: string, extReferenceId?: string) {
    const { error } = await this.client
      .from('video_generations')
      .update({ 
        status,
        extReferenceId
      })
      .eq('id', id)

    if (error) throw new Error(`Database error: ${error.message}`)
  }
}