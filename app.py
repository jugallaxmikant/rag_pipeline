import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding model
# 'all-MiniLM-L6-v2' is a small but powerful model that converts text -> vector embeddings.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#  Initialize Chroma client
client = chromadb.Client()
collection = client.create_collection(name="knowledge_base")

# Load and process documents
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

# Split text into small chunks (because large text can lose context)
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Create embeddings and store them in Chroma
def store_documents(docs):
    ids = []
    texts = []
    embeddings = []

    for i, doc in enumerate(docs):
        chunks = chunk_text(doc)
        for j, chunk in enumerate(chunks):
            chunk_id = f"doc_{i}_chunk_{j}"
            embedding = embedding_model.encode(chunk).tolist()
            ids.append(chunk_id)
            texts.append(chunk)
            embeddings.append(embedding)

    collection.add(ids=ids, embeddings=embeddings, documents=texts)
    print("Stored {len(embeddings)} text chunks in ChromaDB.")


from groq import Groq
import httpx

# Create a custom HTTP client that ignores SSL verification (for testing only)
transport = httpx.HTTPTransport(verify=False)
custom_http_client = httpx.Client(transport=transport)

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Replace the internal HTTP client with the unverified one
groq_client._client = custom_http_client

#  Function: Retrieve top similar chunks
def retrieve_relevant_chunks(query, top_k=3):
    # Convert query to embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Search in ChromaDB for most similar chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extract the matching text chunks
    retrieved_texts = results['documents'][0]
    return retrieved_texts

# Function: Ask question and generate answer using Groq
def generate_answer(query):
    # Retrieve context from stored documents
    context_chunks = retrieve_relevant_chunks(query)
    context = "\n".join(context_chunks)

    # Build the prompt for the Groq LLM
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the context, say "I don't know based on the given documents."

    Context:
    {context}

    Question: {query}
    Answer:
    """

    # Call the Groq LLM
    response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)


    # Extract the model’s text output
    answer = response.choices[0].message.content
    return answer

# CLI interface
if __name__ == "__main__":
    docs = load_documents("docs")
    store_documents(docs)

    print("\n Ask your questions below (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("bye!")
            break

        answer = generate_answer(query)
        print(f"\nAssistant: {answer}\n")
