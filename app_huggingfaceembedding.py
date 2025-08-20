import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Load secrets from .env (Render mounts it in /etc/secrets/.env)
# -------------------------------
if os.path.exists("/etc/secrets/.env"):
    load_dotenv("/etc/secrets/.env")
else:
    load_dotenv()  # fallback for local dev

# -------------------------------
# API Keys
# -------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not HF_TOKEN:
    st.warning("⚠️ HF_TOKEN not found. HuggingFace embeddings may fail.")
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not found. Groq LLM will not work.")

# -------------------------------
# Embeddings & LLM setup
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = None
if GROQ_API_KEY:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based only on the provided context.
    Provide the most accurate response to the question.

    <context>
    {context}
    </context>
    Question: {input}
    """
)

# -------------------------------
# Vector DB creation
# -------------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Load PDFs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 RAG Document Q&A with Groq + HuggingFace")

user_prompt = st.text_input("🔍 Enter your query from the research paper")

if st.button("⚡ Build Document Embedding"):
    create_vector_embedding()
    st.success("✅ Vector Database is ready!")

if user_prompt and llm:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    elapsed = time.process_time() - start

    st.write(f"⏱️ Response time: {elapsed:.2f}s")
    st.write("### Answer:")
    st.success(response['answer'])

    with st.expander("📖 Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---")
elif user_prompt and not llm:
    st.error("❌ LLM not initialized. Please check your GROQ_API_KEY.")

