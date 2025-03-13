import os
import shutil
import time
import requests

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from llama_cpp import Llama  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
os.environ["USER_AGENT"] = USER_AGENT

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ Downloading model from {MODEL_URL}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        
        with open(MODEL_PATH, "wb") as file, open(MODEL_PATH + ".tmp", "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    file.write(chunk)
        
        print("✅ Model download complete!")

download_model()
print("🔄 Loading Mistral model... (This may take a few seconds)")
llm = Llama(
    model_path=MODEL_PATH, 
    n_ctx=2048,         
    n_gpu_layers=40,    
    n_batch=512         
)

print("🔄 Loading Sentence Transformers for FAISS...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DB_FAISS_PATH = "vectorstore/db_faiss"
URL = "https://brainlox.com/courses/category/technical"

def scrape_and_store():
    print("🔄 Scraping website...")

    try:
        loader = WebBaseLoader(URL, headers={"User-Agent": USER_AGENT})
        documents = loader.load()
    except Exception as e:
        print(f"❌ Scraping failed: {str(e)}")
        return {"error": f"Failed to scrape website: {str(e)}"}

    if not documents:
        print("❌ No data scraped. Check the URL or website restrictions.")
        return {"error": "No data scraped."}

    print("🔄 Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.create_documents([doc.page_content for doc in documents])

    print("🔄 Creating FAISS vector store...")
    vector_store = FAISS.from_documents(docs, embeddings)

    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)
    vector_store.save_local(DB_FAISS_PATH)

    print("✅ Vector store created successfully.")
    return {"message": "Website data scraped and stored successfully."}

def generate_response(context, query):
    print(f"📝 Generating response for: {query}")

    trimmed_context = context[:500] if context.strip() else "No relevant data found on the website."

    prompt = f"""You are an AI assistant answering questions based **only on the provided website data**.

### Website Data:
{trimmed_context}

### User Question:
{query}

### AI Response:
"""

    response = llm(
        prompt,
        max_tokens=300,  
        temperature=0.7,
        top_p=0.9
    )

    bot_reply = response["choices"][0]["text"].strip()
    return bot_reply

def stream_response(context, query):
    print(f"📝 Generating streaming response for: {query}")
    
    yield f'{{"message": "AI is thinking...", "typing": true}}\n'
    time.sleep(1)

    response_text = generate_response(context, query)
    
    for word in response_text.split():
        yield f'{{"message": "{word}", "typing": false}}\n'
        time.sleep(0.05)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/scrape', methods=['POST'])
def scrape_data():
    result = scrape_and_store()
    return jsonify(result)

@app.route('/ask', methods=['POST'])
def ask_question():
    if not os.path.exists(DB_FAISS_PATH):
        return jsonify({"error": "No vector store found. Scrape the website first using /scrape"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    query = data["question"].strip()
    if not query:
        return jsonify({"error": "Empty question received"}), 400

    try:
        print("🔄 Loading FAISS vector store...")
        vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return jsonify({"error": f"Failed to load FAISS: {str(e)}"}), 500

    print("🔎 Searching for relevant context...")
    docs = vector_store.similarity_search(query, k=5)
    context = " ".join([doc.page_content for doc in docs])

    return Response(stream_with_context(stream_response(context, query)), content_type='application/json')

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
