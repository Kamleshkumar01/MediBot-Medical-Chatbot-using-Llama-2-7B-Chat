from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY) 
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


from pinecone import Pinecone, ServerlessSpec

# Replace with your actual Pinecone API key
PINECONE_API_KEY = "pcsk_68waxa_TShS9CnuB727QYPGYngU3Wos7nWrtjSdFsYxBWC1tEs1e29RtLehhrrFLVHrGBU"
PINECONE_REGION = "us-east-1"  # Change to your correct region from Pinecone Console
CLOUD_PROVIDER = "aws"  # Use "gcp" if your region is in Google Cloud

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
index_name = "medical-bot"

# Check if index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Set this according to your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD_PROVIDER,
            region=PINECONE_REGION
        )
    )

# Connect to the existing Pinecone index
index = pc.Index(index_name)

print(f"✅ Successfully connected to Pinecone index: {index_name}")



# Convert text chunks into embeddings
embeddings_list = embeddings.embed_documents([t.page_content for t in text_chunks])

# Define batch size (adjust if needed)
BATCH_SIZE = 100  # Process embeddings in chunks of 100

def upsert_in_batches(index, text_chunks, embeddings_list, batch_size=100):
    for i in range(0, len(embeddings_list), batch_size):
        batch_vectors = embeddings_list[i : i + batch_size]  # Get batch
        batch_texts = [t.page_content for t in text_chunks[i : i + batch_size]]  # Store texts
        ids = [str(idx) for idx in range(i, i + len(batch_vectors))]  # Unique IDs

        # Create (ID, vector, metadata) tuples
        vectors = [
            (ids[j], batch_vectors[j], {"text": batch_texts[j]})  # ✅ Store metadata
            for j in range(len(batch_vectors))
        ]

        # Upsert batch into Pinecone
        index.upsert(vectors=vectors)
        print(f"✅ Uploaded batch {i} to {i+len(batch_vectors)} with metadata")

# Call the function
upsert_in_batches(index, text_chunks, embeddings_list, BATCH_SIZE)
print("🚀 Successfully uploaded embeddings with metadata!")
