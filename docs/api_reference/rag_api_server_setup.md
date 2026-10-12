---
title: "RAG API Server - Step by Step Setup Guide"
---

## Prerequisites

### Install Required Dependencies

Python 3.10 or newer is required.

```bash
# FastAPI, uvicorn, PyJWT, python-multipart (gateway) + faiss-cpu, sentence-transformers (rag)
pip install "multimind-sdk[gateway,rag]"

# Optional: local HuggingFace text model (used when no API keys are set)
pip install transformers torch

# Optional: bcrypt password verification for /token
pip install "passlib[bcrypt]"
```

## Step 1: Set API Keys (Optional)

**Note:** By default, the server runs in development mode (no authentication) when neither `API_KEYS` nor `JWT_SECRET` is set. If you have HuggingFace models installed, the server works without provider keys. Provider keys are only needed for OpenAI or Anthropic models.

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

### Option C: Create .env file
Create a file named `.env` in the directory you start the server from:
```
OPENAI_API_KEY=your-actual-openai-api-key-here
# OR
ANTHROPIC_API_KEY=your-actual-anthropic-api-key-here
```

The server loads `.env` from the current working directory automatically (`python-dotenv` is a core dependency, loaded by `multimind.core.config`). Variables already set in the shell take precedence.

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
# Run the server (MUST use -m flag for relative imports to work)
python -m multimind.gateway.rag_api
```

### Method 2: Using uvicorn directly
```bash
uvicorn multimind.gateway.rag_api:app --host 0.0.0.0 --port 8000
```

### Method 3: Combined MultiMind API
```bash
multimind backend            # all REST routes, including these, on http://127.0.0.1:8080/docs
```

**⚠️ IMPORTANT:** Do NOT run `python multimind/gateway/rag_api.py` directly - it will fail with ImportError!

## Step 4: Verify Server is Running

You should see uvicorn start:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The RAG backend is initialized lazily on the first document or query request; at that point the log shows which provider was chosen:
```
INFO:multimind.gateway.rag_api:Checking for API keys
INFO:multimind.gateway.rag_api:OPENAI_API_KEY: Not found
INFO:multimind.gateway.rag_api:ANTHROPIC_API_KEY: Not found
INFO:multimind.gateway.rag_api:RAG system initialized successfully
INFO:multimind.gateway.rag_api:Provider: HuggingFace (Local)
INFO:multimind.gateway.rag_api:Text model: gpt2
INFO:multimind.gateway.rag_api:Embedding model: sentence-transformers/all-MiniLM-L6-v2
INFO:multimind.gateway.rag_api:Vector store: FAISS
```

Open your browser and go to:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health (`/ready` returns 503 until the backend has been initialized)

## Step 5: Test with Client Example

Open a **NEW** terminal window (keep server running in first terminal):

```bash
# From a checkout of the repository (the example is not installed with the package)
cd multimind-sdk

# Run the client example (no API key needed in development mode)
python examples/client/rag_client_example.py
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
   - Text Model: `claude-3-sonnet-20240229`
   - Embeddings: the server still configures OpenAI `text-embedding-ada-002` (dimension 1536) because Anthropic has no embedding models, but without `OPENAI_API_KEY` the embedding client cannot be created and initialization fails with `503 "RAG backend is not available"`.
   - ⚠️ **In practice, Anthropic-only setups need `OPENAI_API_KEY` as well**, or unset `ANTHROPIC_API_KEY` to fall through to the local HuggingFace option and switch the text model afterwards with `POST /models/switch`.

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
- Install the extras: `pip install "multimind-sdk[gateway,rag]"`
- Use `python -m multimind.gateway.rag_api` (with `-m` flag), not direct file execution

### Error: "FAISS is not available"
- Install FAISS: `pip install faiss-cpu` (included in the `rag` extra; or `faiss-gpu` if you have CUDA)
- FAISS is required for the vector store

### Error: "HuggingFace models not available"
- Install: `pip install transformers torch sentence-transformers`
- Models will download automatically on first use (stored in `~/.cache/huggingface/`)

### Error: "Authentication required"
- In development mode, this shouldn't happen if neither `API_KEYS` nor `JWT_SECRET` is set
- If you set `API_KEYS`, you need to provide a valid API key in the client (or a JWT when `JWT_SECRET` is set)
- Remove both variables for development mode

### Error: "JWT authentication is not configured" (503 from `/token`)
- Set `JWT_SECRET` and `JWT_USERS_JSON` (Step 2.1) and install `passlib[bcrypt]`

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
- `GET /health` - Liveness check (no auth)
- `GET /ready` - Readiness check (503 until the RAG backend is initialized)
- `POST /token` - Get JWT token (for authentication)

Full request/response shapes: [rag_api.md](/api_reference/rag_api).

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
- **By default when neither `API_KEYS` nor `JWT_SECRET` is set** → All requests allowed
- **No API keys needed in client** → Works without authentication
- Perfect for local testing and development
- Server automatically uses HuggingFace local models if no provider keys are provided

### Production Mode (With Authentication)
- **Set `API_KEYS` environment variable** → Authentication required
- **Provide API key in client** → `RAGClient(api_key="your-key")`
- Or set `JWT_SECRET` + `JWT_USERS_JSON` and use `/token` bearer tokens (`RAGClient(token=...)` or `await client.login(...)`)
- There is no rate limiting; put the server behind a reverse proxy with TLS

## Summary

1. **Install dependencies**: `pip install "multimind-sdk[gateway,rag]"`
2. **Optional**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (or use local HuggingFace models)
3. **Start server**: `python -m multimind.gateway.rag_api` (or `multimind backend` for the combined API)
4. **Test client**: `python examples/client/rag_client_example.py`
5. **Access docs**: http://localhost:8000/docs

The server automatically detects available models and uses the best option available!
