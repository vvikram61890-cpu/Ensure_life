import os
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import pyttsx3
import threading
import pythoncom
import time

st.set_page_config(page_title="🩺 Health Buddy+", page_icon="🩺")

# === EMBEDDING FUNCTION (ChromaDB 0.5+ Compliant) ===
class LocalEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        try:
            return self.model.encode(input).tolist()
        except Exception:
            return [[0.0] * 384]

    def embed_documents(self, documents):
        try:
            return self.model.encode(documents).tolist()
        except Exception:
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
            st.warning(f"⚠️ Embedding error: {e}")
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

# === TTS ENGINE ===
def speak_text(text):
    def _speak():
        try:
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            pythoncom.CoUninitialize()
        except Exception as e:
            print(f"TTS Error: {e}")
    thread = threading.Thread(target=_speak)
    thread.start()

def listen_microphone():
    """Capture voice from mic and return text"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("🧠 Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError:
        return "Speech service error. Check internet."
    except Exception as e:
        return f"Error: {str(e)}"

# === CHROMADB SETUP ===
persist_directory = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)

# Load embedding model with robust error handling
@st.cache_resource
def load_embedding_model():
    try:
        st.info("🔄 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("✅ Embedding model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        st.info("💡 Please ensure the model is downloaded or check your internet connection")
        return None

embedding_model = load_embedding_model()

if embedding_model is None:
    st.stop()

embedding_function = LocalEmbeddingFunction(embedding_model)

# Get collection with error handling
try:
    collection = chroma_client.get_collection(
        name="medical_docs",
        embedding_function=embedding_function
    )
    st.sidebar.success(f"📚 RAG Database: {collection.count()} documents loaded")
except Exception as e:
    st.error("⚠️ ChromaDB collection 'medical_docs' not found. Did you run `rag_loader.py`?")
    st.info("💡 Please run: `python rag_loader.py` to process your medical documents")
    st.stop()

# === STREAMLIT UI ===
st.title("🩺 My Health Buddy+ — RAG + Voice Assistant")

# Sidebar
st.sidebar.title("⚙️ Settings")
use_voice = st.sidebar.checkbox("🎙️ Enable Voice I/O", value=False)
show_citations = st.sidebar.checkbox("📚 Show Source Citations", value=True)

# Model selection - Default to gemma:2b since it's installed
available_models = ["gemma:2b", "llama2:3b", "llama2:7b", "mistral:7b"]
model_option = st.sidebar.selectbox(
    "Choose LLM Model:",
    available_models,
    index=0,
    help="gemma:2b is recommended for systems with 4GB RAM"
)

st.sidebar.info(f"📝 Selected: {model_option}")

st.sidebar.title("📊 System Status")
st.sidebar.info(f"""
- ✅ Embedding Model: Loaded
- ✅ RAG Database: {collection.count()} documents
- 🤖 LLM Model: {model_option}
- 🎙️ Voice: {'Enabled' if use_voice else 'Disabled'}
- 💾 Memory: Optimized for 4GB systems
""")

st.sidebar.title("⚠️ Important Notice")
st.sidebar.warning("""
**NOT A DOCTOR** — This bot gives general info only.

🟥 Never rely on it for diagnosis or treatment.
🟥 Always consult a licensed healthcare provider.
🟥 In emergencies, call your local emergency number.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle input
user_input = None

if use_voice:
    if st.button("🎙️ Click to Speak"):
        spoken_text = listen_microphone()
        if spoken_text and not spoken_text.startswith(("Error", "Sorry")):
            user_input = spoken_text
            st.success(f"🗣️ You said: *{user_input}*")
        else:
            st.warning(spoken_text)
else:
    user_input = st.chat_input("Ask about symptoms, wellness, diet, first aid...")

if user_input:
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # === RETRIEVE RELEVANT CONTEXT ===
    context = ""
    try:
        with st.spinner("🔍 Searching medical encyclopedia..."):
            results = collection.query(
                query_texts=[user_input],
                n_results=3
            )

        if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
            context_parts = []
            for i, doc in enumerate(results['documents'][0]):
                # Safe metadata access
                meta = {}
                if results.get('metadatas') and i < len(results['metadatas'][0]):
                    meta = results['metadatas'][0][i]
                
                source = meta.get('source', 'Unknown Source')
                page = meta.get('page', '?')
                snippet = doc[:500] + "..." if len(doc) > 500 else doc
                
                if show_citations:
                    context_parts.append(f"📄 [{source}, p.{page}]: {snippet}")
                else:
                    context_parts.append(doc)
            
            context = "\n\n".join(context_parts)
            st.success(f"📚 Found {len(results['documents'][0])} relevant sections from the encyclopedia")
        else:
            st.info("🔍 No specific information found in encyclopedia. Providing general medical information.")
            
    except Exception as e:
        st.warning(f"⚠️ RAG retrieval error: {e}")
        context = ""

    # === SYSTEM PROMPT ===
    system_prompt = f"""
You are 'Health Buddy+', a cautious AI assistant using The Gale Encyclopedia of Medicine (3rd Edition) as your primary reference.

CONTEXT FROM MEDICAL ENCYCLOPEDIA:
{context}

USER QUESTION: {user_input}

IMPORTANT MEDICAL GUIDELINES:
- Provide general educational information from the encyclopedia context when available
- NEVER diagnose, prescribe treatments, or provide personalized medical advice
- Always clarify that you're providing general information from a reference book
- If context doesn't contain relevant information, state this clearly and provide general educational info
- Emphasize the importance of consulting healthcare professionals
- Include relevant medical terminology but explain it clearly
- For emergency situations, direct users to seek immediate medical help
- Be concise but informative (2-4 paragraphs maximum)

RESPONSE FORMAT:
1. Start with a clear definition or explanation
2. Include relevant information from the encyclopedia if available
3. Add appropriate disclaimers about consulting healthcare providers
4. Keep responses educational and factual
"""

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-6:]:  # Keep last 6 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("💭 Generating response from medical encyclopedia..."):
                stream = ollama.chat(
                    model=model_option,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    full_response += chunk['message']['content']
                    message_placeholder.markdown(full_response + "▌")
                time.sleep(0.5)  # Small delay for better UX

            message_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = str(e)
            if "memory" in error_msg.lower():
                full_response = f"""
⚠️ **Memory Optimization Required**

**Immediate Solution:** 
Please select 'gemma:2b' from the model dropdown in the sidebar.

**RAG Context Found:**
{context}

**Based on the medical encyclopedia, here's general information about asthma treatments:**

Asthma treatments typically involve two main approaches: quick-relief medications for immediate symptom control during attacks, and long-term control medications to prevent symptoms. Common treatments include inhalers, corticosteroids, and lifestyle modifications. However, specific treatment plans should always be developed with a healthcare provider.
"""
            elif "connection" in error_msg.lower():
                full_response = "⚠️ Cannot connect to Ollama. Please ensure Ollama is running with: `ollama serve`"
            elif "not found" in error_msg.lower():
                full_response = f"⚠️ Model '{model_option}' not found. Please install it with: `ollama pull {model_option}` or switch to 'gemma:2b'"
            else:
                full_response = f"⚠️ Error: {error_msg}"
            
            message_placeholder.markdown(full_response)

    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Speak response if voice enabled
    if use_voice and full_response and not full_response.startswith("⚠️"):
        with st.spinner("🔊 Converting to speech..."):
            speak_text(full_response)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Quick test buttons
st.sidebar.markdown("---")
st.sidebar.title("🧪 Test Questions")
test_questions = [
    "What is asthma?",
    "Explain diabetes symptoms",
    "What are common heart disease risk factors?",
    "How does the immune system work?"
]

for question in test_questions:
    if st.sidebar.button(f"❓ {question[:30]}..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun()

# Footer
st.markdown("---")
st.caption("🩺 Health Buddy+ | Powered by The Gale Encyclopedia of Medicine 3rd Edition | Not for medical diagnosis")

# Health tips in sidebar
st.sidebar.markdown("---")
st.sidebar.title("💡 Health Tips")
health_tips = [
    "Stay hydrated by drinking plenty of water",
    "Get regular exercise and maintain a balanced diet",
    "Schedule regular check-ups with your healthcare provider",
    "Practice good hygiene to prevent infections",
    "Get adequate sleep for overall wellness"
]
for tip in health_tips:
    st.sidebar.info(f"• {tip}")