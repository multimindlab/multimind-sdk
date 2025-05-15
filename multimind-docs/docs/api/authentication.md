# Authentication

MultiMind SDK supports both API key and JWT authentication for secure access to the RAG API and other services.

## API Key Authentication
- Set your API key as an environment variable (e.g., `OPENAI_API_KEY`).
- Pass the key in your `.env` file or as a header in API requests.

## JWT Authentication
- Obtain a JWT token via the `/token` endpoint.
- Pass the token as a Bearer token in the `Authorization` header.

## Security Best Practices
- Never commit API keys or tokens to version control.
- Use environment variables for secrets.
- Rotate keys regularly and use least-privilege principles.

_See the [Quickstart](../getting-started/quickstart.md) for setup examples._ 