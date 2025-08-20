import streamlit as st
import os
import time
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
if os.path.exists("/etc/secrets/.env"):
    load_dotenv("/etc/secrets/.env")
else:
    load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# LangChain imports
# -------------------------------
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -------------------------------
# Initialize embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------
# Initialize LLMs
# -------------------------------
available_llms = {}

if groq_api_key:
    available_llms["Groq Llama3"] = ChatGroq(
        groq_api_key=groq_api_key, model_name="Llama3-8b-8192"
    )

if openai_key:
    available_llms["OpenAI GPT-4o-mini"] = ChatOpenAI(
        openai_api_key=openai_key, model="gpt-4o-mini"
    )

if hf_token:
    available_llms["HuggingFace Llama-2"] = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.6, "max_length": 512},
    )

# -------------------------------
# Prompt Template
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response to the question.

    <context>
    {context}
    </context>
    Question: {input}
    """
)

# -------------------------------
# Vector DB Creation
# -------------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 Multi-Provider RAG Q&A (Groq | OpenAI | HuggingFace)")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("⚡ Build Document Embedding"):
    create_vector_embedding()
    st.success("✅ Vector Database is ready!")

if user_prompt and available_llms:
    # Create tabs for each LLM provider
    tabs = st.tabs(list(available_llms.keys()))

    for (name, llm), tab in zip(available_llms.items(), tabs):
        with tab:
            st.subheader(f"🔮 {name}")
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_prompt})
            elapsed = time.process_time() - start

            st.write(f"⏱️ Response time: {elapsed:.2f}s")
            st.success(response["answer"])

            with st.expander("📖 Document similarity search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("---")

elif user_prompt and not available_llms:
    st.error("❌ No LLMs available. Please set your API keys in Render Secret Files.")
