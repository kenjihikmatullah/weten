
## Weten (Experimental)
Weten is an experimental project to create short video using totally-free tools

Stack
- Video footage: Pexels
- Voice script generator: OpenRouter with llama-4-maverick
- Text-to-speech : Coqui-AI TTS
- Video assemble: ffmpeg
- AI workflow: n8n
- Containerization: Docker

### Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kenjihikmatullah/weten.git
    cd weten
    ```

2.  **Set environment variables:**
    
    Create .env file, for example
    ```bash
    N8N_PORT=5678
    N8N_BASIC_AUTH_ACTIVE=true
    N8N_BASIC_AUTH_USER=admin
    N8N_BASIC_AUTH_PASSWORD=password

    ```

3.  **Build and run the Docker containers:**
    ```bash
    docker-compose up --build -d
    ```

4.  **Access n8n:**
    Open your web browser and navigate to `http://localhost:5678`

5.  **Configure Pexels API Key in n8n:**
    Once n8n is running, you will need to add your Pexels API key as a credential within the n8n interface.


## Usage

After setting up the project, you can import the `workflow.json` file into your n8n instance. This workflow orchestrates the process of fetching images from Pexels, generating voice scripts with OpenRouter, converting text to speech with Coqui-AI TTS, and finally compiling the video with FFmpeg.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or find bugs

