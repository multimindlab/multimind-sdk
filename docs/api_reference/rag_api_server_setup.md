# RAG API Server - Step by Step Setup Guide

## Prerequisites

### Install Required Dependencies

```bash
# Install core dependencies
pip install PyJWT fastapi uvicorn

# Install FAISS for vector store (required)
pip install faiss-cpu
# OR if you have CUDA GPU:
# pip install faiss-gpu

# Optional: For local HuggingFace models (if no API keys)
pip install transformers torch sentence-transformers
```

## Step 1: Set API Keys (Optional)

**Note:** By default, the server runs in development mode when `API_KEYS` is not set. If you have HuggingFace models installed, the server works without API keys. API keys are only needed for OpenAI or Anthropic models.

### Option A: Windows PowerShell (Temporary - Current Session Only)
```powershell
# For OpenAI
$env:OPENAI_API_KEY="your-actual-openai-api-key-here"

# OR for Anthropic
$env:ANTHROPIC_API_KEY="your-actual-anthropic-api-key-here"
```

### Option B: Windows Command Prompt (Temporary - Current Session Only)
```cmd
# For OpenAI
set OPENAI_API_KEY=your-actual-openai-api-key-here

# OR for Anthropic
set ANTHROPIC_API_KEY=your-actual-anthropic-api-key-here
```

### Option C: Create .env file (Requires python-dotenv)
Create a file named `.env` in your project root:
```
OPENAI_API_KEY=your-actual-openai-api-key-here
# OR
ANTHROPIC_API_KEY=your-actual-anthropic-api-key-here
```

**⚠️ Important:** The server does NOT automatically load `.env` files. To use a `.env` file:
1. Install python-dotenv: `pip install python-dotenv`
2. Load it in your code before running the server, OR
3. Use environment variables directly (Option A or B above)

**Note:** If you don't set any API keys, the server will automatically use local HuggingFace models (requires `transformers` and `torch` installed).

## Step 2: Verify API Key is Set (If Using API Keys)

### In PowerShell:
```powershell
echo $env:OPENAI_API_KEY
# OR
echo $env:ANTHROPIC_API_KEY
```

### In Command Prompt:
```cmd
echo %OPENAI_API_KEY%
# OR
echo %ANTHROPIC_API_KEY%
```

You should see your API key printed. If nothing shows, the server will use local HuggingFace models.

## Step 2.1: Configure JWT/Auth Security Variables (`.env`)

If you want to use `/token` JWT auth (production-style), add these values to your `.env`.
This matches `.env.example` (`API_KEYS`, `JWT_SECRET`, `JWT_USERS_JSON`).

### Required/Optional fields

- `API_KEYS` (optional): comma-separated API keys for `X-API-Key` auth
- `JWT_SECRET` (required for `/token`): long random secret (32+ chars)
- `JWT_USERS_JSON` (required for `/token`): JSON mapping username -> bcrypt hash

### 1) Generate `JWT_SECRET`

PowerShell:
```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy output into:
```env
JWT_SECRET=your_generated_secret_here
```

### 2) Generate bcrypt password hash

Install passlib+bcrypt (if needed):
```bash
pip install passlib[bcrypt]
```

Generate hash:
```powershell
python -c "from passlib.context import CryptContext; c=CryptContext(schemes=['bcrypt'], deprecated='auto'); print(c.hash('YourStrongPassword123!'))"
```

### 3) Set `JWT_USERS_JSON`

Use the generated hash in JSON (single line):
```env
JWT_USERS_JSON={"admin":"$2b$12$replace_with_bcrypt_hash","testuser":"$2b$12$replace_with_bcrypt_hash"}
```

### 4) Optional `API_KEYS`

```env
API_KEYS=key_one,key_two,key_three
```

### Example secure `.env` block

```env
OPENAI_API_KEY=your-openai-key
API_KEYS=internal_key_1,internal_key_2
JWT_SECRET=replace_with_long_random_secret_32plus_chars
JWT_USERS_JSON={"admin":"$2b$12$replace_with_bcrypt_hash"}
```

**Important:**
- Do not commit real secrets to git.
- Do not use default/fallback secrets in production.
- If `JWT_SECRET`/`JWT_USERS_JSON` are missing, `/token` auth should be considered not securely configured.

## Step 3: Start the RAG API Server

### Method 1: Run as module (RECOMMENDED)
```bash
# Navigate to your project directory (where multimind-sdk is located)
cd your-project-directory\multimind-sdk

# Run the server (MUST use -m flag for relative imports to work)
python -m multimind.gateway.rag_api
```

### Method 2: Using uvicorn directly
```bash
uvicorn multimind.gateway.rag_api:app --host 0.0.0.0 --port 8000
```

**⚠️ IMPORTANT:** Do NOT run `python multimind\gateway\rag_api.py` directly - it will fail with ImportError!

## Step 4: Verify Server is Running

You should see output like:
```
📋 Checking for API keys...
   OPENAI_API_KEY: ❌ Not found
   ANTHROPIC_API_KEY: ❌ Not found

============================================================
✅ RAG SYSTEM INITIALIZED SUCCESSFULLY
============================================================
📊 Provider: HuggingFace (Local)
📝 Text Model: gpt2
🔤 Embedding Model: sentence-transformers/all-MiniLM-L6-v2
📦 Vector Store: FAISS
============================================================

INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open your browser and go to:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Step 5: Test with Client Example

Open a **NEW** terminal window (keep server running in first terminal):

```bash
# Navigate to your project directory (where multimind-sdk is located)
cd your-project-directory\multimind-sdk

# Run the client example (no API key needed in development mode)
python examples\client\rag_client_example.py
```

**Note:** In development mode, you can remove or set `api_key=None` in the client:
```python
client = RAGClient(
    base_url="http://localhost:8000"
    # No api_key needed in development mode
)
```

## Model Selection Priority

The server automatically selects models in this order:

1. **OpenAI** (if `OPENAI_API_KEY` is set)
   - Text Model: `gpt-3.5-turbo`
   - Embedding Model: `text-embedding-ada-002`
   - Dimension: 1536

2. **Anthropic** (if `ANTHROPIC_API_KEY` is set, but no OpenAI key)
   - Text Model: `claude-3-sonnet-20240229` ✅ (Anthropic provides this)
   - Embeddings: 
     - ⚠️ **Anthropic does NOT provide embedding models**
     - Requires one of the following:
       - **OpenAI embeddings** (`text-embedding-ada-002`) - needs `OPENAI_API_KEY`
       - **OR HuggingFace local embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) - no API key needed
   - Dimension: 1536 (if using OpenAI) or 384 (if using HuggingFace)
   - ⚠️ **Important:** Anthropic only provides text generation, not embeddings

3. **HuggingFace Local** (if no API keys are set)
   - Text Model: `gpt2`
   - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Dimension: 384
   - ✅ No API keys required - runs completely locally
   - Models download automatically on first use

## Troubleshooting

### Error: "No model API keys found"
- **Solution 1:** Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable
- **Solution 2:** Install HuggingFace models: `pip install transformers torch sentence-transformers`
- Restart your terminal after setting environment variables
- Verify with `echo $env:OPENAI_API_KEY` (PowerShell) or `echo %OPENAI_API_KEY%` (CMD)

### Error: "Cannot connect to host localhost:8000"
- Make sure the server is running (Step 3)
- Check if port 8000 is already in use
- Try accessing http://localhost:8000/health in browser
- Make sure you're using the correct URL in the client

### Error: "Module not found" or "ImportError"
- Make sure you're in the project directory
- Install dependencies: `pip install PyJWT fastapi uvicorn`
- Use `python -m multimind.gateway.rag_api` (with `-m` flag), not direct file execution

### Error: "FAISS is not available"
- Install FAISS: `pip install faiss-cpu` (or `faiss-gpu` if you have CUDA)
- FAISS is required for the vector store

### Error: "HuggingFace models not available"
- Install: `pip install transformers torch sentence-transformers`
- Models will download automatically on first use (stored in `~/.cache/huggingface/`)

### Error: "Authentication required"
- In development mode, this shouldn't happen if `API_KEYS` environment variable is not set
- If you set `API_KEYS`, you need to provide a valid API key in the client
- Remove `API_KEYS` environment variable for development mode

## API Endpoints

Once running, you can access:
- **Swagger UI (Interactive Docs)**: http://localhost:8000/docs
- **ReDoc (Alternative Docs)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Available Endpoints

- `POST /documents` - Add documents to the RAG system
- `POST /files` - Upload and add files
- `POST /query` - Query documents
- `POST /generate` - Generate responses using RAG
- `DELETE /documents` - Clear all documents
- `GET /documents/count` - Get document count
- `POST /models/switch` - Switch text generation model
- `GET /health` - Health check
- `POST /token` - Get JWT token (for authentication)

## Example API Calls

### Using curl (PowerShell):
```powershell
# Health check
curl http://localhost:8000/health

# Add documents
curl -X POST http://localhost:8000/documents `
  -H "Content-Type: application/json" `
  -d '{\"documents\":[{\"text\":\"Test document\",\"metadata\":{}}]}'

# Query
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"What is the RAG system?\",\"top_k\":3}'

# Generate response
curl -X POST http://localhost:8000/generate `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"Explain the RAG system\",\"temperature\":0.7}'
```

## Development Mode vs Production Mode

### Development Mode (No Authentication)
- **By default when `API_KEYS` environment variable is NOT set** → All requests allowed
- **No API keys needed in client** → Works without authentication
- Perfect for local testing and development
- Server automatically uses HuggingFace local models if no API keys are provided

### Production Mode (With Authentication)
- **Set `API_KEYS` environment variable** → Authentication required
- **Provide API key in client** → `RAGClient(api_key="your-key")`
- Use JWT tokens for more secure authentication

## Summary

1. **Install dependencies**: `pip install PyJWT fastapi uvicorn faiss-cpu`
2. **Optional**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (or use local HuggingFace models)
3. **Start server**: `python -m multimind.gateway.rag_api`
4. **Test client**: `python examples\client\rag_client_example.py`
5. **Access docs**: http://localhost:8000/docs

The server automatically detects available models and uses the best option available!
