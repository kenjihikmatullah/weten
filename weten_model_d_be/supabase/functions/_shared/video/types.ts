export interface GenerateVideoRequest {
  prompt: string;
  resolution: string;
  duration: number;
  form: string;
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