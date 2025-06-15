// Generate video
export interface GenerateVideoRequest {
  prompt: string;
  resolution?: string;
  duration?: number;
  form?: string;
}

export interface GenerateVideoResponse {
    success: boolean;
    message: string;
    videoGenerationId?: string;
}

export interface VideoGeneration {
  id?: number;
  userId: string;
  status: 'TO_PROCESS' | 'PROCESSING' | 'DONE' | 'FAILED';
  userPrompt: string;
  resolution: string;
  duration: number;
  extReferenceId?: string;
  form: string;
  createdAt: string;
  updatedAt: string;
}

// Get video list
export interface VideoListResponse {
  video_generation_id: number;
  video_id?: number;
  status: string;
  url?: string;
  title?: string;
  description?: string;
  prompt: string;
  resolution: string;
  duration: number;
  form: string;
  startedGenerationAt: string;
  finishedGenerationAt?: string;
}