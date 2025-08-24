# Streamlit Secrets Management

This project uses Streamlit's native secrets management for secure credential storage.

## Setup

1. **Create the secrets file:**

   ```bash
   # Copy the example
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. **Edit `.streamlit/secrets.toml` with your actual values:**

   ```toml
   # Root-level secrets (available as environment variables and via st.secrets)
   DEPLOYMENT_MODE = "cloud"
   GROQ_API_KEY = "your_actual_groq_api_key"
   PINECONE_API_KEY = "your_actual_pinecone_api_key"
   HUGGINGFACE_API_TOKEN = "your_actual_huggingface_token"

   # Grouped secrets
   [pinecone]
   environment = "gcp-starter"
   index_name = "migracion"

   [groq]
   model = "llama3-8b-8192"

   [huggingface]
   embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
   ```

3. **The secrets file is automatically ignored by git** (already in `.gitignore`)

## Usage

The app automatically loads secrets from `.streamlit/secrets.toml` and makes them available as:

- **st.secrets dictionary**: `st.secrets["GROQ_API_KEY"]` or `st.secrets.pinecone.environment`
- **Environment variables**: Root-level secrets are also available as `os.environ["GROQ_API_KEY"]`

## Fallback

If no secrets file exists, the app falls back to `.env` file and environment variables.

## Deployment

For **Streamlit Community Cloud**, add your secrets through the web interface:

1. Go to your app settings
2. Navigate to "Secrets"
3. Add the same TOML content there

## Local Development

- Never commit `.streamlit/secrets.toml` to git
- Use `.streamlit/secrets.toml.example` as a template
- Root-level secrets automatically become environment variables
- Nested sections can be accessed via `st.secrets.section.key`
