# Atlas 🎬

> AI-Powered Content Analysis Platform for Educational and Research Content

Atlas is a comprehensive platform that combines YouTube video analysis, academic paper research, and educational content generation into a unified AI-powered workflow.

## Features

### 🔍 YouTube Pipeline
- **Video Search**: Natural language search using YouTube Data API
- **Transcript Extraction**: Dual-method fetching (youtube-transcript-api + yt-dlp fallback)
- **AI Summarization**: Technical content analysis with structured insights
- **Comparison Analysis**: Multi-video comparison with AI-powered insights

### 📚 Academic RAG System
- **Semantic Search**: Query academic papers using natural language
- **Citation Tracking**: Source papers with relevance scores
- **Vector Database**: LanceDB-powered semantic search
- **Paper Management**: Automatic PDF processing and indexing

### 📝 Educational Content
- **Assignment Generation**: AI-created hands-on learning exercises
- **Learning Objectives**: Structured educational outcomes
- **Progressive Tasks**: Step-by-step skill building activities
- **Assessment Criteria**: Clear success metrics and rubrics

### ⚡ Advanced Processing
- **Parallel Execution**: Concurrent processing for faster results
- **Real-time Tracking**: Progressive visualization of pipeline steps
- **Professional Interface**: Modern web UI with responsive design
- **Configurable Workers**: Adjustable concurrency for optimal performance

### 🔐 Authentication & Access Control
- **Google OAuth**: Secure authentication via Google accounts
- **Role-based Access**: Restrict features to allowed users
- **BYOK Support**: Bring Your Own Keys for non-privileged users
- **History Tracking**: Pipeline run history with user attribution

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key
- YouTube Data API key
- MySQL database (optional, for history tracking)

### Installation

```bash
git clone https://github.com/sudipta26889/atlas
cd atlas
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

```bash
# Create .env file with required variables
cat > .env << EOF
OPENAI_API_KEY=your_openai_key
YOUTUBE_API_KEY=your_youtube_key

# Optional: MySQL for history tracking
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=atlas_db

# Optional: Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Optional: Access control (comma-separated emails)
ALLOWED_HISTORY_EMAILS=admin@example.com,user@example.com
EOF
```

### Launch Atlas

```bash
python app.py
```

Access the web interface at `http://localhost:7860`

### Docker Deployment

```bash
# Build and run with Docker
docker build -t atlas .
docker run -p 7860:7860 --env-file .env atlas
```

Or use the pre-built image from GitHub Container Registry:
```bash
docker pull ghcr.io/sudipta26889/atlas:latest
docker run -p 7860:7860 --env-file .env ghcr.io/sudipta26889/atlas:latest
```

## Usage

### 1. YouTube Analysis
1. Enter a search query (e.g., "Python machine learning tutorial")
2. Configure max videos and workers
3. Click "Start Pipeline" to begin processing
4. View results: search → transcripts → summaries → comparison → assignments

### 2. Academic Papers Query
1. Ensure papers are in `papers/agents/` folder
2. Enter natural language query
3. Get AI responses with paper citations and excerpts

### 3. History (Admin Only)
- View past pipeline runs
- Re-run previous searches
- Delete old runs and output files

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for summarization |
| `YOUTUBE_API_KEY` | Yes | YouTube Data API key for search |
| `MYSQL_HOST` | No | MySQL host for history tracking |
| `MYSQL_PORT` | No | MySQL port (default: 3306) |
| `MYSQL_USER` | No | MySQL username |
| `MYSQL_PASSWORD` | No | MySQL password |
| `MYSQL_DATABASE` | No | MySQL database name |
| `GOOGLE_CLIENT_ID` | No | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | No | Google OAuth client secret |
| `ALLOWED_HISTORY_EMAILS` | No | Comma-separated list of admin emails |

### Config File

Key settings in `src/configs/config.yaml`:
- **Model**: OpenAI model selection
- **Workers**: Parallel processing configuration
- **API**: Timeout and retry settings
- **Paths**: Output directories and file locations

## Project Structure

```
atlas/
├── app.py                          # Main Gradio web interface
├── Dockerfile                      # Docker build configuration
├── requirements.txt                # Python dependencies
├── src/
│   ├── youtube_pipeline.py         # YouTube processing pipeline
│   ├── fetch_youtube_transcript.py # Transcript fetching (API + yt-dlp)
│   ├── papers_rag.py               # Academic papers RAG system
│   ├── assignment_generator.py     # Educational content generator
│   ├── compare_youtube_outputs.py  # Video comparison analysis
│   ├── database.py                 # MySQL history tracking
│   └── configs/config.yaml         # Configuration settings
├── papers/agents/                  # Academic papers directory
└── .github/workflows/              # CI/CD workflows
```

## Architecture

### Transcript Fetching
Atlas uses a dual-method approach for reliable transcript extraction:
1. **Primary**: `youtube-transcript-api` - Fast, no authentication required
2. **Fallback**: `yt-dlp` - Handles edge cases, supports cookies for bot bypass

### Authentication Flow
1. User accesses the app → Redirected to Google OAuth
2. After login → Email checked against `ALLOWED_HISTORY_EMAILS`
3. Allowed users → Full access including env API keys
4. Other users → BYOK (Bring Your Own Keys) mode

## License

MIT License - See LICENSE file for details
