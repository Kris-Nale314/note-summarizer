# Transcript Analyzer Deployment Guide

This guide provides instructions for setting up and deploying the Transcript Analyzer application both locally and using Docker.

## Local Development Setup

### Prerequisites
- Python 3.11 or later
- pip (Python package manager)
- An OpenAI API key

### Setting Up the Environment

1. **Clone or download the repository**

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

5. **Download NLTK data**:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

6. **Run the application**:
   ```bash
   streamlit run app.py
   ```
   
   Alternatively, use the provided script:
   ```bash
   chmod +x run_local.sh
   ./run_local.sh
   ```

7. **Access the application** at http://localhost:8501

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- An OpenAI API key

### Building and Running with Docker

1. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

2. **Build and start the container**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application** at http://localhost:8501

4. **Stop the container** when finished:
   ```bash
   docker-compose down
   ```

### Docker Configuration Notes

- The application will be exposed on port 8501
- A volume is mounted at `/app/data` for data persistence
- Your `.env` file is mounted into the container for configuration

## Production Deployment Considerations

For production deployments, consider the following:

1. **Security**:
   - Secure your API keys using secrets management
   - Consider using HTTPS with a reverse proxy (Nginx, Traefik, etc.)
   - Implement proper authentication if needed

2. **Scaling**:
   - The application can be resource-intensive when processing large transcripts
   - Consider deploying on a machine with at least 4 CPU cores and 8GB RAM

3. **Monitoring**:
   - Add logging to monitor application performance
   - Consider implementing health checks for container orchestration

4. **OpenAI API Costs**:
   - Be aware that the application makes multiple API calls to OpenAI
   - Set up proper budget monitoring for your OpenAI API usage

## File Structure

```
transcript-analyzer/
├── app.py              # Streamlit application
├── crew.py             # CrewAI implementation
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── .env.example        # Example environment variables
├── .env                # Your environment variables (create from example)
├── data/               # Directory for persistent data
└── run_local.sh        # Script to run locally
```

## Troubleshooting

1. **OpenAI API Key Issues**:
   - Ensure your API key is correctly set in the `.env` file
   - Check that you have sufficient credits in your OpenAI account

2. **Docker Issues**:
   - If port 8501 is already in use, modify the port mapping in `docker-compose.yml`
   - For permission issues, ensure Docker has proper access to the mounted volumes

3. **Memory Issues**:
   - If the application crashes when processing large transcripts, try reducing the number of chunks
   - Consider increasing the memory available to the Docker container

4. **Missing Dependencies**:
   - If you encounter missing dependency errors, ensure all packages in `requirements.txt` are installed
   - Some packages may require additional system dependencies on certain platforms