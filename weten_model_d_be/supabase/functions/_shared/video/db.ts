import { createClient, SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2.39.0'
import { VideoGeneration, VideoListResponse } from './types.ts'

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

  async checkAndUpdateCompletedVideos(userId: string) {
    // Get incomplete video generations
    const { data: generations, error: genError } = await this.client
      .from('video_generations')
      .select('*')
      .eq('userId', userId)
      .neq('status', 'DONE');

    if (genError) throw new Error(`Database error: ${genError.message}`);
    if (!generations?.length) return;

    for (const gen of generations) {
      try {
        // Check if video exists in storage
        const { data: fileExists } = await this.client
          .storage
          .from('videos')
          .list('', {
            limit: 1,
            search: `${gen.id}.mp4`
          });

        if (fileExists && fileExists.length > 0) {
          // Get file URL
          const { data: { publicUrl: url } } = this.client
            .storage
            .from('videos')
            .getPublicUrl(`${gen.id}.mp4`);

          // Create video record
          const { error: vidError } = await this.client
            .from('videos')
            .insert({
              videoGenerationId: gen.id,
              title: `Generated Video ${gen.id}`,
              description: gen.userPrompt,
              url
            });

          if (vidError) throw vidError;

          // Update generation status
          await this.updateVideoGenerationStatus(gen.id, 'DONE');
        }
      } catch (error) {
        console.error(`Error processing generation ${gen.id}:`, error);
      }
    }
  }

  async getVideosList(userId: string): Promise<VideoListResponse[]> {
    // Get incomplete videos from video_generations
    const { data: generations, error: genError } = await this.client
      .from('video_generations')
      .select('*')
      .eq('userId', userId)
      .neq('status', 'DONE')
      .order('createdAt', { ascending: false });

    if (genError) throw new Error(`Database error: ${genError.message}`);

    // Get completed videos
    const { data: videos, error: vidError } = await this.client
      .from('videos')
      .select(`
        *,
        video_generations!inner(
          id,
          userPrompt,
          resolution,
          duration,
          form,
          createdAt
        )
      `)
      .eq('video_generations.userId', userId)
      .order('createdAt', { ascending: false });

    if (vidError) throw new Error(`Database error: ${vidError.message}`);

    // Format incomplete videos
    const incompleteVideos: VideoListResponse[] = generations.map(gen => ({
      video_generation_id: gen.id,
      status: gen.status,
      prompt: gen.userPrompt,
      resolution: gen.resolution,
      duration: gen.duration,
      form: gen.form,
      startedGenerationAt: gen.createdAt
    }));

    // Format completed videos
    const completedVideos: VideoListResponse[] = videos.map(vid => ({
      video_generation_id: vid.video_generations.id,
      video_id: vid.id,
      status: 'DONE',
      url: vid.url,
      title: vid.title,
      description: vid.description,
      prompt: vid.video_generations.userPrompt,
      resolution: vid.video_generations.resolution,
      duration: vid.video_generations.duration,
      form: vid.video_generations.form,
      startedGenerationAt: vid.video_generations.createdAt,
      finishedGenerationAt: vid.createdAt
    }));

    // Combine and sort by startedGenerationAt
    return [...incompleteVideos, ...completedVideos]
      .sort((a, b) => new Date(b.startedGenerationAt).getTime() - new Date(a.startedGenerationAt).getTime());
  }
}