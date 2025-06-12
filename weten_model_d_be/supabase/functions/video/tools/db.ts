import { createClient, SupabaseClient } from '@supabase/supabase-js'
import { VideoGeneration } from './types'

export class DbClient {
  private client: SupabaseClient

  constructor() {
    this.client = createClient(
      process.env.SUPABASE_URL ?? '',
      process.env.SUPABASE_SERVICE_ROLE_KEY ?? ''
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