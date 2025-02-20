import os
import logging
from flask import Flask, render_template, request
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_core.prompts import PromptTemplate  
from langchain.chains import RetrievalQA  
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  
from langchain_pinecone import PineconeVectorStore  # ✅ Use updated Pinecone class
from langchain_community.llms import CTransformers
from src.prompt import prompt_template  

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = "us-east-1"  
CLOUD_PROVIDER = "aws"  

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing. Please check your .env file.")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone client
try:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
except Exception as e:
    logger.error(f"❌ Failed to initialize Pinecone client: {e}")
    raise


# Define index name 
index_name = "medical-bot"

# Check if index exists, if not, create it
try:
    if index_name not in pc.list_indexes().names():
        logger.info(f"🔍 Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=PINECONE_REGION)
        )

    # Connect to the existing Pinecone index
    index = pc.Index(index_name)
    logger.info(f"✅ Successfully connected to Pinecone index: {index_name}")

except Exception as e:
    logger.error(f"❌ Error connecting to Pinecone: {e}")
    raise

# Load the existing Pinecone index correctly
docsearch = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")  # ✅ Use updated class

# Define prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load Llama model
try:
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.7}
    )
except Exception as e:
    logger.error(f"❌ Failed to load Llama model: {e}")
    raise

# Define QA Retrieval Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),  # ✅ Fixed incorrect argument
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    
    if not user_input or not user_input.strip():
        logger.warning("⚠️ Empty user input received.")
        return "⚠️ Please enter a valid question."

    logger.info(f"User Input: {user_input}")

    try:
        # Use .invoke() instead of .run() (LangChain API update)
        response = qa.invoke({"query": user_input})  # ✅ Fixed LangChain deprecation warning

        # Extract only the response text (ignore source documents)
        result_text = response.get("result", "⚠️ No response generated.")

        logger.info(f"Response: {result_text}")
        return str(result_text)

    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        return "⚠️ An error occurred. Please try again."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
