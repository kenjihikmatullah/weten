export interface RunpodInput {
  prompt: string
  resolution: string
  duration: number
}

export class RunpodClient {
  private endpoint: string
  private apiKey: string

  constructor() {
    this.endpoint = Deno.env.get('RUNPOD_API_URL') ?? ''
    this.apiKey = Deno.env.get('RUNPOD_API_KEY') ?? ''
  }

  async generateVideo(input: RunpodInput): Promise<string> {
    const response = await fetch(`${this.endpoint}/run`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input })
    })

    if (!response.ok) {
      throw new Error(`Runpod API error: ${response.statusText}`)
    }

    const data = await response.json()
    return data.id
  }
}