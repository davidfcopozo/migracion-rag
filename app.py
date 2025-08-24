import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Prefer the newer langchain-chroma package (avoids LangChain deprecation warnings).
# Fall back to the older import if the new package isn't installed to preserve compatibility.
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import time
import json
import re
from dotenv import load_dotenv

# Load Streamlit secrets and make them available as environment variables
# Streamlit secrets are automatically loaded from .streamlit/secrets.toml
def _load_streamlit_secrets():
    try:
        # Access st.secrets to ensure they're loaded, but don't fail if no secrets file exists
        if hasattr(st, 'secrets') and len(st.secrets) > 0:
            secrets_loaded = 0
            # Make root-level secrets available as environment variables (if not already set)
            for key, value in st.secrets.items():
                if isinstance(value, (str, int, float, bool)) and key.upper() not in os.environ:
                    os.environ[key.upper()] = str(value)
                    secrets_loaded += 1
                    
            # Handle nested sections by flattening them
            for section_key, section_value in st.secrets.items():
                if isinstance(section_value, dict):
                    for nested_key, nested_value in section_value.items():
                        env_key = f"{section_key.upper()}_{nested_key.upper()}"
                        if env_key not in os.environ and isinstance(nested_value, (str, int, float, bool)):
                            os.environ[env_key] = str(nested_value)
                            secrets_loaded += 1
            
            # Log successful loading (but don't expose secret values)
            if secrets_loaded > 0:
                import logging
                logging.info(f"Loaded {secrets_loaded} secrets from st.secrets into environment")
                
    except Exception as e:
        # st.secrets may not be available yet or no secrets file exists
        # Fall back to .env loading
        import logging
        logging.info(f"st.secrets not available ({e}), falling back to .env")

# Load Streamlit secrets first, then .env as fallback
_load_streamlit_secrets()

# Load environment variables from .env (keeps already-set envs intact)
load_dotenv()

# Make HF token available under common env names and try to login early so Streamlit
# worker subprocesses see the credentials. This reduces 401s caused by env propagation.
_hf_token = os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HF_TOKEN')
if _hf_token:
    os.environ.setdefault('HUGGINGFACEHUB_API_TOKEN', _hf_token)
    os.environ.setdefault('HUGGINGFACE_API_TOKEN', _hf_token)
    os.environ.setdefault('HF_TOKEN', _hf_token)
    try:
        from huggingface_hub import login as _hf_login
        try:
            _hf_login(_hf_token)
        except Exception:
            # Non-fatal; env vars are usually sufficient
            pass
    except Exception:
        # huggingface_hub not installed or login not available; continue
        pass

# Configuration based on deployment mode
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local").lower()

# Cloud imports (only when needed)
if DEPLOYMENT_MODE == "cloud":
    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
    except ImportError as e:
        st.error(f"Cloud dependencies not installed: {e}")
        st.stop()

# Chat history persistence
CHAT_HISTORY_FILE = "./chat_history.json"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            logging.warning("Failed to load chat history file; starting fresh.")
    return None

def save_chat_history(messages):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save chat history: {e}")

def format_response_progressive(text: str) -> str:
    """Apply progressive formatting suitable for streaming display.
    
    Uses formatting that doesn't change text size during streaming.
    """
    if not text:
        return ""
    
    # Clean basic whitespace
    text = text.strip()
    
    # Add a simple bold header without markdown headers (# ##) to avoid size changes
    if not text.startswith('**') and len(text) > 10:
        text = f"**📋 Información**\n\n{text}"
    
    # Format numbered lists in real-time (conservative approach)
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append(line)
            continue
            
        # Look for numbered list patterns and format them gently
        if ':' in line and re.search(r'\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ].*?\s+\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]', line):
            parts = line.split(':', 1)
            if len(parts) == 2:
                intro = parts[0].strip() + ':'
                content = parts[1].strip()
                
                # Split numbered items
                items = re.split(r'\s+(\d+)\s+', content)
                if len(items) > 2:
                    formatted_lines.append(f"**{intro}**")  # Bold instead of header
                    formatted_lines.append("")
                    
                    for i in range(1, len(items), 2):
                        if i + 1 < len(items):
                            num = items[i]
                            item_text = items[i + 1].strip()
                            if item_text:
                                formatted_lines.append(f"• **{num}.** {item_text}")  # Bullet + bold number
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines).strip()

def format_response_md(text: str) -> str:
    """Post-process model text to produce well-structured Markdown.
    
    Simple, safe formatting that preserves word integrity.
    """
    if not text:
        return ""

    # Clean basic whitespace
    text = text.strip()
    
    # Only add header if response doesn't already have structure
    if not text.startswith('#') and not text.startswith('**') and not text.startswith('##'):
        text = f"## 📋 Información\n\n{text}"
    
    # Simple fix for obvious numbered lists: "item: 1 text 2 text" 
    # Split into sentences and look for patterns within complete sentences
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append(line)
            continue
            
        # Look for pattern: ends with colon, followed by "1 word ... 2 word"
        if ':' in line and re.search(r'\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ].*?\s+\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]', line):
            # Split at the colon
            parts = line.split(':', 1)
            if len(parts) == 2:
                intro = parts[0].strip() + ':'
                content = parts[1].strip()
                
                # Split numbered items (very conservative approach)
                items = re.split(r'\s+(\d+)\s+', content)
                if len(items) > 2:  # Has actual numbered content
                    formatted_lines.append(intro)
                    formatted_lines.append("")  # Empty line
                    
                    # Process items: [text_before_1, "1", text_1, "2", text_2, ...]
                    for i in range(1, len(items), 2):
                        if i + 1 < len(items):
                            num = items[i]
                            item_text = items[i + 1].strip()
                            if item_text:
                                formatted_lines.append(f"{num}. {item_text}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    # Rejoin and clean up
    text = '\n'.join(formatted_lines)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Add footer
    text += "\n\n---\n*Basado en la Ley de Migración de la República Dominicana*"
    
    return text.strip()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants and helpers
DOC_PATH = "./data/LEY-DE-MIGRACION.pdf"  # default; we'll fall back to any PDF in ./data

def get_default_pdf_path():
    """Return a usable PDF path: prefer DOC_PATH if exists; else first *.pdf under ./data."""
    try:
        if os.path.exists(DOC_PATH):
            return DOC_PATH
        data_dir = os.path.join(".", "data")
        if os.path.isdir(data_dir):
            for name in os.listdir(data_dir):
                if name.lower().endswith(".pdf"):
                    path = os.path.join(data_dir, name)
                    if os.path.isfile(path):
                        logging.info(f"Using discovered PDF: {path}")
                        return path
    except Exception as e:
        logging.warning(f"Error discovering PDF files: {e}")
    return None

def format_docs(docs):
    """Join document contents for the prompt context."""
    try:
        parts = []
        for i, d in enumerate(docs, 1):
            content = getattr(d, "page_content", str(d))
            parts.append(f"[Fragment {i}]\n{content}")
        return "\n\n".join(parts)
    except Exception:
        return ""

# Configuration based on deployment mode
if DEPLOYMENT_MODE == "local":
    # Local configuration (Ollama + ChromaDB)
    MODEL_NAME = "llama3.2"
    EMBEDDING_MODEL_NAME = "nomic-embed-text"
    VECTOR_STORE_NAME = "ley-de-migracion-rag"
    PERSIST_DIRECTORY = "./local_dbs/migracion_chroma_db"
else:
    # Cloud configuration (Groq + Pinecone + HuggingFace)
    # Helper function to safely get values from st.secrets or environment
    def get_config_value(key, section_key=None, nested_key=None, default=None):
        """Get configuration value from st.secrets or environment variables"""
        # Try st.secrets first
        try:
            # Try root-level secret
            if key in st.secrets:
                return st.secrets[key]
            # Try nested section if provided
            if section_key and nested_key and section_key in st.secrets:
                section = st.secrets[section_key]
                if isinstance(section, dict) and nested_key in section:
                    return section[nested_key]
        except Exception:
            pass
        # Fall back to environment variable
        return os.getenv(key, default)
    
    MODEL_NAME = get_config_value("GROQ_MODEL", "groq", "model", "llama3-8b-8192")
    EMBEDDING_MODEL_NAME = get_config_value("HUGGINGFACE_EMBEDDING_MODEL", "huggingface", "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    PINECONE_INDEX_NAME = get_config_value("PINECONE_INDEX_NAME", "pinecone", "index_name", "migracion")
    PINECONE_ENVIRONMENT = get_config_value("PINECONE_ENVIRONMENT", "pinecone", "environment", "gcp-starter")
    
    # API Keys validation
    GROQ_API_KEY = get_config_value("GROQ_API_KEY")
    PINECONE_API_KEY = get_config_value("PINECONE_API_KEY")
    HUGGINGFACE_API_TOKEN = get_config_value("HUGGINGFACE_API_TOKEN")
    
    if not all([GROQ_API_KEY, PINECONE_API_KEY, HUGGINGFACE_API_TOKEN]):
        st.error("Cloud mode requires GROQ_API_KEY, PINECONE_API_KEY, and HUGGINGFACE_API_TOKEN. "
                "Set them in .streamlit/secrets.toml or as environment variables.")
        st.stop()
    
    # Debug: Show configuration source (remove in production)
    with st.expander("🔧 Configuration Debug", expanded=False):
        st.write("**Deployment Mode:**", DEPLOYMENT_MODE)
        st.write("**Model Name:**", MODEL_NAME)
        st.write("**Embedding Model:**", EMBEDDING_MODEL_NAME)
        st.write("**Pinecone Index:**", PINECONE_INDEX_NAME)
        st.write("**Pinecone Environment:**", PINECONE_ENVIRONMENT)
        st.write("**API Keys Present:**")
        st.write(f"- GROQ_API_KEY: {'✅' if GROQ_API_KEY else '❌'}")
        st.write(f"- PINECONE_API_KEY: {'✅' if PINECONE_API_KEY else '❌'}")
        st.write(f"- HUGGINGFACE_API_TOKEN: {'✅' if HUGGINGFACE_API_TOKEN else '❌'}")
        
        # Show if values came from st.secrets or environment
        st.write("**Configuration Sources:**")
        try:
            sources = {}
            for key, display_name in [
                ("GROQ_MODEL", "Model"),
                ("PINECONE_INDEX_NAME", "Index"),
                ("PINECONE_ENVIRONMENT", "Environment"),
                ("GROQ_API_KEY", "Groq Key"),
                ("PINECONE_API_KEY", "Pinecone Key"),
                ("HUGGINGFACE_API_TOKEN", "HF Token")
            ]:
                source = "❓"
                try:
                    if key in st.secrets:
                        source = "🔐 st.secrets (root)"
                    elif key in os.environ:
                        source = "🌍 environment"
                    else:
                        # Check nested sections
                        for section_name in ["groq", "pinecone", "huggingface"]:
                            if section_name in st.secrets and isinstance(st.secrets[section_name], dict):
                                section = st.secrets[section_name]
                                nested_key = key.lower().replace(f"{section_name}_", "").replace("_api_key", "").replace("_token", "")
                                if nested_key in section:
                                    source = f"🔐 st.secrets.{section_name}"
                                    break
                except Exception:
                    pass
                st.write(f"- {display_name}: {source}")
        except Exception as e:
            st.write(f"Could not determine sources: {e}")

def ensure_model_available(model_name):
    """Ensure model availability depending on deployment mode.
    - local: pull Ollama model
    - cloud: assume remote model availability
    Returns True if usable, False otherwise.
    """
    if DEPLOYMENT_MODE == "cloud":
        # Remote models (Groq) are managed by provider; no local pull needed
        return True
    try:
        logging.info(f"Ensuring Ollama model is available: {model_name}")
        ollama.pull(model_name)
        logging.info(f"Model {model_name} is available (pulled successfully).")
        return True
    except Exception as e:
        logging.warning(f"Failed to pull model {model_name}: {e}")
        return False

# 1. Ingest PDF files
def ingest_pdf(doc_path):
    """Ingest a PDF file and return its contents."""
    if doc_path:
        loader = UnstructuredPDFLoader(doc_path)
        data = loader.load()
        logging.info("PDF ingestion complete.")
        return data
    else:
        logging.warning("No document path provided.")
        st.error("PDF file not found.")
        return None

# 2. Extract text from PDF file and split into chunks

def split_documents(data):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)  # data returned from step 1

    logging.info("PDF splitting and chunking complete.")
    return chunks

# 3. Send chunks to embedding model and store embeddings in vector database
@st.cache_resource
def load_vector_db():
    """Load or create the vector database based on deployment mode."""
    if DEPLOYMENT_MODE == "local":
        return load_local_vector_db()
    else:
        return load_cloud_vector_db()

def load_local_vector_db():
    """Load or create ChromaDB vector database for local deployment."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL_NAME)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing local vector database.")
    else:
        # Load and process the PDF document
        doc_path = get_default_pdf_path()
        if not doc_path:
            st.error("No se encontró ningún PDF en ./data. Agrega un archivo .pdf y vuelve a intentar.")
            return None
        data = ingest_pdf(doc_path)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        # Persist the vector DB if the implementation provides a persist method.
        try:
            if hasattr(vector_db, "persist"):
                vector_db.persist()
                logging.info("Vector database created and persisted via .persist().")
            # Some implementations manage persistence via an internal client
            elif hasattr(vector_db, "client") and hasattr(vector_db.client, "persist"):
                vector_db.client.persist()
                logging.info("Vector database persisted via vector_db.client.persist().")
            else:
                # Newer langchain-chroma implementations may persist automatically or expose
                # a different API; log this so the developer can inspect.
                logging.info(
                    "Vector database created; no persist() method found. "
                    "Persistence may be automatic for this backend."
                )
        except Exception as e:
            logging.warning(f"Could not persist vector database: {e}")

    return vector_db

def load_cloud_vector_db():
    """Load or create Pinecone vector database for cloud deployment."""
    # Fallback local embeddings implementation (uses sentence-transformers or transformers+torch)
    class LocalTransformersEmbeddings:
        def __init__(self, model_name=EMBEDDING_MODEL_NAME):
            self.model_name = model_name
            self._method = None
            # Try sentence-transformers first (easier API)
            try:
                from sentence_transformers import SentenceTransformer

                self._smodel = SentenceTransformer(model_name)
                self._method = "sentence_transformers"
            except Exception:
                # Fallback to transformers + torch
                try:
                    from transformers import AutoTokenizer, AutoModel
                    import torch

                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._tmodel = AutoModel.from_pretrained(model_name)
                    self._torch = torch
                    self._method = "transformers"
                except Exception as e:
                    raise RuntimeError(
                        f"Could not load a local embedding model for '{model_name}': {e}.\n"
                        "Install 'sentence-transformers' or 'transformers'+'torch' to use local embeddings."
                    )

        def embed_documents(self, texts):
            if not texts:
                return []
            if self._method == "sentence_transformers":
                embs = self._smodel.encode(texts, convert_to_numpy=True)
                return [list(map(float, e)) for e in embs]
            # transformers + torch path
            all_embs = []
            for t in texts:
                inputs = self._tokenizer(t, truncation=True, padding=True, return_tensors="pt")
                with self._torch.no_grad():
                    model_output = self._tmodel(**inputs)
                token_embeddings = model_output.last_hidden_state
                attention_mask = inputs["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
                sum_mask = input_mask_expanded.sum(1)
                sentence_embedding = sum_embeddings / sum_mask
                emb = sentence_embedding[0].cpu().numpy()
                all_embs.append(list(map(float, emb)))
            return all_embs

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    # Initialize embeddings
    # Ensure HF token is available under common env names used by libraries
    if HUGGINGFACE_API_TOKEN:
        os.environ.setdefault('HUGGINGFACEHUB_API_TOKEN', HUGGINGFACE_API_TOKEN)
        os.environ.setdefault('HUGGINGFACE_API_TOKEN', HUGGINGFACE_API_TOKEN)
        os.environ.setdefault('HF_TOKEN', HUGGINGFACE_API_TOKEN)

    # Ensure env vars are present for libraries that read them
    if HUGGINGFACE_API_TOKEN:
        os.environ.setdefault('HUGGINGFACEHUB_API_TOKEN', HUGGINGFACE_API_TOKEN)
        os.environ.setdefault('HUGGINGFACE_API_TOKEN', HUGGINGFACE_API_TOKEN)
        os.environ.setdefault('HF_TOKEN', HUGGINGFACE_API_TOKEN)

    # Optionally call huggingface_hub.login() to register token for libraries that support it
    try:
        from huggingface_hub import login as hf_login
        try:
            hf_login(HUGGINGFACE_API_TOKEN)
        except Exception:
            # non-fatal; env vars should suffice
            pass
    except Exception:
        # huggingface_hub not available or login not needed; continue
        pass

    # Instantiate embeddings (do not pass unknown kwargs)
    embedding = None
    backend = ""
    try:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        backend = "huggingface_hub"
    except Exception as e_first:
        logging.warning(f"HuggingFaceEmbeddings with auth failed ({e_first}). Trying anonymous access for public models.")
        # Try anonymous by temporarily removing HF auth env vars
        prev_env = {
            k: os.environ.get(k)
            for k in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HF_TOKEN")
        }
        try:
            for k in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HF_TOKEN"):
                if k in os.environ:
                    del os.environ[k]
            embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            backend = "huggingface_hub (anonymous)"
        except Exception as e_anon:
            logging.warning(f"Anonymous HuggingFaceEmbeddings failed ({e_anon}). Falling back to local transformers.")
            # Restore env before fallback
            for k, v in prev_env.items():
                if v is not None:
                    os.environ[k] = v
            # Try a local fallback that doesn't call the Hugging Face API (requires local model download)
            try:
                embedding = LocalTransformersEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                backend = "local_transformers"
                st.warning(
                    "HuggingFaceEmbeddings unavailable (401/auth). Using local 'transformers' fallback for embeddings. "
                    "Install 'sentence-transformers' or 'transformers'+'torch' if you want this to work."
                )
            except Exception as e_local:
                logging.error(f"Failed to create local fallback embeddings: {e_local}")
                raise
        else:
            # Restore env after successful anonymous init
            for k, v in prev_env.items():
                if v is not None:
                    os.environ[k] = v

    # Determine embedding dimension dynamically
    try:
        test_vec = embedding.embed_query("dim-check")
        embed_dim = len(test_vec) if hasattr(test_vec, "__len__") else 384
    except Exception:
        embed_dim = 384
    logging.info(f"Embeddings backend: {backend or 'unknown'}, dimension: {embed_dim}")
    st.caption(f"Embeddings: {backend or 'unknown'} · dim {embed_dim}")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Helpers to get indexes list and dimensions
    def _list_index_names_and_dims():
        names = []
        dims = {}
        try:
            listed = pc.list_indexes()
            if hasattr(listed, 'names') and callable(listed.names):
                names = listed.names()
                # Try to also pull dimensions if available in models
                try:
                    for item in listed:
                        n = getattr(item, 'name', None) or (item.get('name') if isinstance(item, dict) else None)
                        d = getattr(item, 'dimension', None) if not isinstance(item, dict) else item.get('dimension')
                        if n and d:
                            dims[n] = d
                except Exception:
                    pass
            elif isinstance(listed, (list, tuple)):
                for i in listed:
                    if hasattr(i, 'name'):
                        n = i.name
                        names.append(n)
                        d = getattr(i, 'dimension', None)
                        if d:
                            dims[n] = d
                    elif isinstance(i, dict) and 'name' in i:
                        names.append(i['name'])
                        if 'dimension' in i:
                            dims[i['name']] = i['dimension']
                    elif isinstance(i, str):
                        names.append(i)
        except Exception:
            pass
        return names, dims

    def _describe_index_dim(name):
        try:
            desc = pc.describe_index(name)
            if isinstance(desc, dict):
                return desc.get('dimension') or desc.get('index_config', {}).get('dimension')
            return getattr(desc, 'dimension', None) or getattr(getattr(desc, 'index_config', None), 'dimension', None)
        except Exception:
            return None

    index_names, index_dims = _list_index_names_and_dims()

    # Choose effective index name matching the embedding dimension
    effective_index_name = PINECONE_INDEX_NAME
    existing_dim = None
    if PINECONE_INDEX_NAME in index_names:
        existing_dim = index_dims.get(PINECONE_INDEX_NAME) or _describe_index_dim(PINECONE_INDEX_NAME)
        if existing_dim and existing_dim != embed_dim:
            # Dimension mismatch: route to a suffixed index name to avoid 400 errors
            suffixed = f"{PINECONE_INDEX_NAME}-{embed_dim}"
            st.warning(
                f"El índice '{PINECONE_INDEX_NAME}' existe con dimensión {existing_dim}, "
                f"pero el modelo de embeddings produce {embed_dim}. Usando '{suffixed}'."
            )
            effective_index_name = suffixed

    if effective_index_name in index_names:
        logging.info(f"Using existing Pinecone index: {effective_index_name}")
        # Check if index is empty; if so, populate it from the PDF
        try:
            index = pc.Index(effective_index_name)
            stats = index.describe_index_stats()
            # Support dict- or object-style
            total_vectors = None
            if isinstance(stats, dict):
                total_vectors = stats.get("total_vector_count") or stats.get("totalVectors")
            else:
                total_vectors = getattr(stats, "total_vector_count", None) or getattr(stats, "totalVectors", None)
        except Exception as e:
            logging.warning(f"Could not fetch Pinecone index stats: {e}")
            total_vectors = None

        if total_vectors is None:
            logging.info("Index stats unavailable; probing index with a similarity search.")
            vector_db = PineconeVectorStore.from_existing_index(
                index_name=effective_index_name,
                embedding=embedding
            )
            try:
                probe = vector_db.similarity_search("initialization probe", k=1)
            except Exception as probe_e:
                logging.warning(f"Probe search failed, proceeding with existing index: {probe_e}")
                probe = None
            if isinstance(probe, list) and len(probe) == 0:
                logging.info("Probe indicates empty index; ingesting PDF and populating the index.")
                st.info("Inicializando la base vectorial en Pinecone (primer uso)...")
                doc_path = get_default_pdf_path()
                if not doc_path:
                    st.error("No se encontró ningún PDF en ./data. Agrega un archivo .pdf y vuelve a intentar.")
                    return None
                data = ingest_pdf(doc_path)
                if data is None:
                    return None
                chunks = split_documents(data)
                try:
                    vector_db = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embedding,
                        index_name=effective_index_name
                    )
                    logging.info("Pinecone index populated successfully (probe path).")
                except Exception as up_e:
                    logging.error(f"Failed to populate Pinecone index after probe: {up_e}")
                    st.error("No se pudo inicializar la base vectorial en Pinecone. Revisa las credenciales y vuelve a intentar.")
                    return None
        elif total_vectors <= 0:
            logging.info("Pinecone index is empty; ingesting PDF and populating the index.")
            st.info("Inicializando la base vectorial en Pinecone (primer uso)...")
            # Load and process the PDF document
            doc_path = get_default_pdf_path()
            if not doc_path:
                st.error("No se encontró ningún PDF en ./data. Agrega un archivo .pdf y vuelve a intentar.")
                return None
            data = ingest_pdf(doc_path)
            if data is None:
                return None
            chunks = split_documents(data)
            try:
                vector_db = PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embedding,
                    index_name=effective_index_name
                )
                logging.info("Pinecone index populated successfully.")
            except Exception as up_e:
                logging.error(f"Failed to populate existing Pinecone index: {up_e}")
                st.error("No se pudo inicializar la base vectorial en Pinecone. Revisa las credenciales y vuelve a intentar.")
                return None
        else:
            logging.info(f"Pinecone index has {total_vectors} vectors; using it.")
            vector_db = PineconeVectorStore.from_existing_index(
                index_name=effective_index_name,
                embedding=embedding
            )
    else:
        logging.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Load and process the PDF document
        doc_path = get_default_pdf_path()
        if not doc_path:
            st.error("No se encontró ningún PDF en ./data. Agrega un archivo .pdf y vuelve a intentar.")
            return None
        data = ingest_pdf(doc_path)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        # Create the index (Pinecone free tier supports up to 1536 dimensions)
        try:
            # Prefer new ServerlessSpec if available
            from pinecone import ServerlessSpec
            pc.create_index(
                name=effective_index_name,
                dimension=embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception:
            # Fallback to dict-based spec if SDK is older
            pc.create_index(
                name=effective_index_name,
                dimension=embed_dim,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )
        
        # Create vector store from documents
        vector_db = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            index_name=effective_index_name
        )
        
        logging.info("Pinecone vector database created successfully.")

    return vector_db

# 4. Perform similarity search on the vector database to find relevant documents

# set up retriever
def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""

    QUERY_PROMPT = PromptTemplate(input_variables=["question"], template= """Tu eres un asistente de IA especializado en la ley de migración de la República Dominicana. Tu tarea es generar diferentes versiones de la consulta del usuario para obtener documentos relevantes desde la base de datos vectorial. Al generar diferentes versiones de la consulta, tu cometido es ayudar al usuario a superar las limitaciones de la búsqueda de similitud basada en la distancia. Provee estas preguntas alternativas separadas por nuevas líneas. La consulta original es: {question}""")

    # 5. Retrieve relevant documents and display results
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(), # set in step 3
        llm=llm, # set in step 4
        prompt=QUERY_PROMPT,
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(llm):
    """Create the prompt->LLM chain; supply context+question at invoke time."""

    # RAG template - encourage structured responses
    template = """Responde la consulta basado SOLAMENTE en el siguiente contexto. 
    
Organiza tu respuesta de manera clara y estructurada:
- Si hay varios puntos, enuméralos claramente (1., 2., 3., etc.)
- Si hay pasos o requisitos, preséntalos como una lista numerada
- Usa párrafos separados para diferentes temas
- Incluye detalles específicos del contexto cuando sea relevante

Contexto:
{context}

Pregunta: {question}

Respuesta estructurada:"""

    prompt = ChatPromptTemplate.from_template(template)

    # retriever is the relevant documents we got from the vector store

    # the chain will pass the context (retriever)
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created with preserved syntax.")
    return chain

def main():
    st.title("Consultor de la Ley de Migración de la República Dominicana")
    st.caption(f"Modo: {'🌐 Cloud (Groq + Pinecone + HF)' if DEPLOYMENT_MODE=='cloud' else '💻 Local (Ollama + Chroma)'}")

    # User input
    #user_input = st.text_input("Haz una pregunta:", "")
    # Initialize chat history from persistent store if available
    if "messages" not in st.session_state:
        persisted = load_chat_history()
        if persisted:
            st.session_state.messages = persisted
        else:
            st.session_state.messages = [{"role": "assistant", "content": "¡Haz una pregunta! 👇"}]

    # UI control: clear chat
    if st.button("Limpiar chat"):
        st.session_state.messages = [{"role": "assistant", "content": "¡Haz una pregunta! 👇"}]
        try:
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
        except Exception:
            logging.warning("Could not delete chat history file.")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("¿Cuál es tu consulta?")

    if user_input:
        # Immediately show the user's message in the chat and record it in history
        with st.chat_message("user"):
            # Render the user's input as markdown so any formatting is preserved
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_chat_history(st.session_state.messages)

        with st.spinner("Consultando la ley de migración..."):
            try:
                # Initialize the language model depending on mode
                if DEPLOYMENT_MODE == "cloud":
                    # Groq LLM (reads GROQ_API_KEY from env)
                    llm = ChatGroq(model=MODEL_NAME)
                else:
                    # Ensure the Ollama model is available locally and init
                    if not ensure_model_available(MODEL_NAME):
                        st.error(
                            f"Model \"{MODEL_NAME}\" not found and could not be pulled.\n"
                            "Make sure Ollama is running and the model name is correct, or pull it manually using the ollama CLI."
                        )
                        return
                    llm = ChatOllama(model=MODEL_NAME)

                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain (takes context+question)
                chain = create_chain(llm)

                # Retrieve docs for this question
                docs = retriever.get_relevant_documents(user_input)
                if not docs:
                    st.warning("No se encontró contexto relevante en la base vectorial. Verifica el PDF en ./data o reformula la pregunta.")
                    return

                context_str = format_docs(docs)

                # Get the response (may be plain text)
                response = chain.invoke({"context": context_str, "question": user_input})

                # Get raw response text for streaming (without formatting)
                raw_response = str(response)

                # Stream the response into the chat with progressive formatting
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    displayed_text = ""
                    words = raw_response.split()
                    
                    for i, word in enumerate(words):
                        displayed_text += word + " "
                        time.sleep(0.03)
                        
                        # Apply progressive formatting that maintains text size
                        formatted_progressive = format_response_progressive(displayed_text)
                        
                        # Show typing indicator with progressive formatting
                        message_placeholder.markdown(formatted_progressive + " ▌")
                    
                    # Apply final formatting and render result
                    formatted_response = format_response_md(raw_response)
                    message_placeholder.markdown(formatted_response)

                # Save formatted response to chat history
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                save_chat_history(st.session_state.messages)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Por favor, ingresa una consulta.")

# Run the main function
if __name__ == "__main__":
    main()
