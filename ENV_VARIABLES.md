# Environment Variables and Configuration

This document explains how the app loads and uses configuration values from environment variables and Streamlit secrets.

## Configuration Priority

The app uses the following priority order for configuration values:

1. **Streamlit secrets** (`.streamlit/secrets.toml`) - highest priority
2. **Environment variables** - medium priority
3. **Default values** - lowest priority

## Environment Variable Loading Process

### 1. Streamlit Secrets Loading

```python
# Loads from .streamlit/secrets.toml
_load_streamlit_secrets()
```

- Root-level secrets become environment variables: `GROQ_API_KEY` → `os.environ["GROQ_API_KEY"]`
- Nested sections are flattened: `pinecone.environment` → `os.environ["PINECONE_ENVIRONMENT"]`

### 2. .env File Loading

```python
# Loads from .env file (preserves existing env vars)
load_dotenv()
```

### 3. Configuration Resolution

```python
# Smart config loading with fallbacks
def get_config_value(key, section_key=None, nested_key=None, default=None):
    # Try st.secrets first (root level)
    # Try st.secrets nested sections
    # Fall back to environment variables
    # Return default if nothing found
```

## Required Environment Variables (Cloud Mode)

| Variable                | Source Options  | Example      |
| ----------------------- | --------------- | ------------ |
| `DEPLOYMENT_MODE`       | `.env`, secrets | `"cloud"`    |
| `GROQ_API_KEY`          | secrets, env    | `"gsk_..."`  |
| `PINECONE_API_KEY`      | secrets, env    | `"pcsk_..."` |
| `HUGGINGFACE_API_TOKEN` | secrets, env    | `"hf_..."`   |

## Optional Configuration Variables

| Variable                      | Default                                    | Description          |
| ----------------------------- | ------------------------------------------ | -------------------- |
| `GROQ_MODEL`                  | `"llama3-8b-8192"`                         | Groq model name      |
| `PINECONE_INDEX_NAME`         | `"migracion"`                              | Pinecone index name  |
| `PINECONE_ENVIRONMENT`        | `"gcp-starter"`                            | Pinecone environment |
| `HUGGINGFACE_EMBEDDING_MODEL` | `"sentence-transformers/all-MiniLM-L6-v2"` | HF embedding model   |

## Configuration Sources Debug

When running in cloud mode, the app shows a "🔧 Configuration Debug" panel that displays:

- Current configuration values
- Source of each configuration (secrets vs environment)
- API key presence indicators

## Example Configurations

### .streamlit/secrets.toml

```toml
DEPLOYMENT_MODE = "cloud"
GROQ_API_KEY = "gsk_your_key_here"
PINECONE_API_KEY = "pcsk_your_key_here"
HUGGINGFACE_API_TOKEN = "hf_your_token_here"

[pinecone]
environment = "gcp-starter"
index_name = "migracion"

[groq]
model = "llama3-8b-8192"
```

### .env

```bash
DEPLOYMENT_MODE=cloud
GROQ_API_KEY=gsk_your_key_here
PINECONE_API_KEY=pcsk_your_key_here
HUGGINGFACE_API_TOKEN=hf_your_token_here
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=migracion
GROQ_MODEL=llama3-8b-8192
```

## Validation

The app validates that all required API keys are present before proceeding:

```python
if not all([GROQ_API_KEY, PINECONE_API_KEY, HUGGINGFACE_API_TOKEN]):
    st.error("Missing required API keys...")
    st.stop()
```

## Troubleshooting

1. **Missing API keys**: Check the debug panel to see which keys are missing
2. **Wrong values**: Verify `.streamlit/secrets.toml` and `.env` files
3. **Secrets not loading**: Restart Streamlit after changing `secrets.toml`
4. **Priority issues**: Remember secrets override environment variables
