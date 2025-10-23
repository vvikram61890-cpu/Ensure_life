import os
import warnings
# Suppress warnings and enable offline mode
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import sys

class LocalEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        try:
            embeddings = self.model.encode(input)
            if len(embeddings.shape) == 1:
                embeddings = [embeddings]
            result = []
            for vec in embeddings:
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                if isinstance(vec, (float, int)):
                    vec = [float(vec)]
                elif hasattr(vec, '__iter__'):
                    vec = [float(x) for x in vec]
                else:
                    vec = [0.0]
                result.append(vec)
            return result
        except Exception as e:
            print(f"⚠️ Embedding error in __call__: {e}")
            return [[0.0] * 384]

    def embed_documents(self, documents):
        try:
            embeddings = self.model.encode(documents)
            result = []
            for vec in embeddings:
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                if isinstance(vec, (float, int)):
                    vec = [float(vec)]
                elif hasattr(vec, '__iter__'):
                    vec = [float(x) for x in vec]
                else:
                    vec = [0.0]
                result.append(vec)
            return result
        except Exception as e:
            print(f"⚠️ Embedding error in embed_documents: {e}")
            return [[0.0] * 384] * len(documents)

    def embed_query(self, input):
        # Handle both positional and keyword arguments
        if isinstance(input, dict) and 'input' in input:
            query = input['input']
        else:
            query = input
            
        # Handle both string and list inputs
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query
            
        # Encode the query/queries
        try:
            embeddings = self.model.encode(queries)
            
            # Convert to list format
            if len(embeddings.shape) == 1:
                embedding_list = [embeddings.tolist()]
            else:
                embedding_list = embeddings.tolist()
                
            return embedding_list
            
        except Exception as e:
            print(f"⚠️ Embedding error in embed_query: {e}")
            if isinstance(query, str):
                return [[0.0] * 384]
            else:
                return [[0.0] * 384] * len(query)

    def __str__(self):
        return "LocalEmbeddingFunction"

    def __repr__(self):
        return "LocalEmbeddingFunction"

    def name(self):
        return "LocalEmbeddingFunction"

def load_pdfs_and_embed(folder_path="./data"):
    documents = []
    metadatas = []
    ids = []

    if not os.path.exists(folder_path):
        print(f"📁 Folder '{folder_path}' not found. Creating it...")
        os.makedirs(folder_path)
        print(f"✅ Please add PDF files to '{folder_path}' and re-run this script.")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDFs found in '{folder_path}'. Add medical/health PDFs and re-run.")
        return

    print(f"📄 Found {len(pdf_files)} PDF(s): {pdf_files}")

    total_pages = 0
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        print(f"  → Processing: {filename}")
        try:
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(text.strip())
                    metadatas.append({"source": filename, "page": i + 1})
                    ids.append(f"{filename}_page{i+1}")
                    total_pages += 1
            print(f"    ✅ Extracted {len(reader.pages)} pages")
        except Exception as e:
            print(f"    ❌ Error reading {filename}: {e}")

    if not documents:
        print("❌ No valid text extracted from PDFs.")
        return

    print(f"📖 Total pages with text: {total_pages}")
    print(f"🧠 Embedding {len(documents)} document chunks...")

    # Initialize model and embedding function
    try:
        print("🔄 Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        print("💡 Please check your internet connection or run:")
        print("   python -c \"from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')\"")
        return

    embedding_function = LocalEmbeddingFunction(embedding_model)

    # Initialize ChromaDB
    persist_directory = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # Create or get collection
    try:
        # Try to get existing collection first
        collection = chroma_client.get_collection(
            name="medical_docs",
            embedding_function=embedding_function
        )
        print("📚 Using existing collection 'medical_docs'")
        
        # Clear existing data to avoid duplicates
        collection.delete(where={})
        print("🧹 Cleared existing documents from collection")
        
    except Exception:
        # Create new collection if it doesn't exist
        collection = chroma_client.create_collection(
            name="medical_docs",
            embedding_function=embedding_function
        )
        print("📚 Created new collection 'medical_docs'")

    # Add to DB
    try:
        print("💾 Adding documents to database...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ SUCCESS: Loaded & embedded {len(documents)} document chunks from {len(pdf_files)} PDF(s).")
        print(f"📊 Collection now contains {collection.count()} documents.")
        
    except Exception as e:
        print(f"❌ Error adding documents to database: {e}")


if __name__ == "__main__":
    print("🚀 Starting RAG Database Setup...")
    load_pdfs_and_embed()
    print("🎉 RAG setup completed!")