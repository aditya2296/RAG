import pdfplumber
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings

# Load the PDF and extract text
pdf_path = "swift.pdf"

def clean_table_data(df):
    # Strip whitespaces and handle empty cells
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Replace common artifacts or missing values
    df.replace(
        {"—": "No", "—": "N/A", "(cid:859)": "Yes", "": "N/A", None: "N/A"}, 
        inplace=True
    )

    # Drop entirely empty rows
    df.dropna(how="all", inplace=True)

    # Ensure correct column data types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except:
            pass

    return df

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

text_data = extract_text(pdf_path)

# Extract tables
def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])  # First row as header
                df = clean_table_data(df)
                tables.append(df)
    return tables

tables = extract_tables(pdf_path)

# Combine all tables into one DataFrame
combined_df = pd.concat(tables, ignore_index=True)

# Save to CSV
combined_df.to_csv("extracted_tables_combined.csv", index=False)

print("Saved all tables to 'extracted_tables_combined.csv'")

# from langchain.schema import Document

# # Use a character-based recursive splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# # Apply chunking to text data
# text_chunks = text_splitter.split_text(text_data)

# # Convert into Documents
# text_documents = [Document(page_content=chunk) for chunk in text_chunks]

# # Convert tables into text format for embedding
# def row_to_text(row, columns):
#     return " | ".join([f"{col}: {val}" for col, val in zip(columns, row)])

# table_texts = []
# for df in tables:
#     table_texts.extend(df.apply(lambda row: row_to_text(row, df.columns), axis=1).tolist())

# # Load sentence-transformers model for embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Compute embeddings
# row_embeddings = embedding_model.encode(table_texts)

# from sklearn.cluster import KMeans
# # Optimal number of clusters dynamically
# num_clusters = max(2, len(row_embeddings) // 10)
# # Use K-Means for better table clustering
# clustering = KMeans(n_clusters=num_clusters, random_state=42).fit(row_embeddings)

# # Group rows based on clusters
# semantic_chunks = {}
# for i, label in enumerate(clustering.labels_):
#     if label not in semantic_chunks:
#         semantic_chunks[label] = []
#     semantic_chunks[label].append(table_texts[i])

# # Convert chunks into text format
# semantic_table_chunks = ["\n".join(rows) for rows in semantic_chunks.values()]
# table_documents = [Document(page_content=chunk) for chunk in semantic_table_chunks]

# # Use HuggingFace embeddings for local processing (no API required)
# embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Store both text and table chunks in FAISS
# vector_store = FAISS.from_documents(text_documents + table_documents, embedding_function)

# # Save FAISS index for later retrieval
# vector_store.save_local("faiss_index")

# print("Chunking complete! Data stored in FAISS for retrieval.")

# # Load stored FAISS index
# vector_store = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)

# # Set your Gemini 2.0 Flash API Key
# API_KEY = "AIzaSyC0AYzALz-d4_SuUW9OqtXSzfu3uWCgBnw"
# genai.configure(api_key=API_KEY)

# # Load Gemini 2.0 Flash Model
# def generate_answer_with_gemini(query, context):
#     """
#     Use Gemini 2.0 Flash to generate a context-aware answer.
#     """
#     prompt = (
#     f"Answer the following query with relevant and factual information from the context provided.\n"
#     f"Query: {query}\n\n"
#     f"Context:\n{context}\n\n"
#     f"Instructions:\n"
#     f"- Prioritize information based on relevance.\n"
#     f"- If context is insufficient, indicate that.\n"
#     f"- Be concise but comprehensive in the answer.\n")

#     # Configure model parameters
#     model = genai.GenerativeModel(model_name="gemini-2.0-flash")

#     # Generate response from Gemini 2.0 Flash
#     response = model.generate_content(prompt)

#     # Extract and return the generated answer
#     if response.candidates and response.candidates[0].content:
#         answer = response.candidates[0].content.parts[0].text.strip()
#         return answer
#     else:
#         return "❗ Unable to generate an answer. Please try again."

# def retrieve_and_answer_with_gemini(query, top_k=5):
#     """
#     Enhanced retrieval and answer generation using Gemini 2.0 Flash.
#     """
#     # Retrieve relevant chunks from FAISS
#     results = vector_store.similarity_search(query, k=top_k)
#     retrieved_texts = [result.page_content for result in results]

#     # Combine top-k retrieved results as context for Gemini
#     context = "\n\n".join(retrieved_texts)

#     # Generate final answer using Gemini 2.0 Flash
#     answer = generate_answer_with_gemini(query, context)
    
#     print(f"\n🔍 **Final Answer:**\n{answer}")
#     return answer

# # Example Queries
# queries = [
#     "Does Lxi model of swift has LED Projector Headlamps?",
#     "What are the Gasoline variants available of Swift?",
#     "What is the seating capacity of swift?",  # Should return a number
#     "Which all models of swift has LED Projector Headlamps?",
#     "What is the length width and height of swift?"
# ]

# for query in queries:
#     print("\n📝 Query:", query)
#     retrieve_and_answer_with_gemini(query)