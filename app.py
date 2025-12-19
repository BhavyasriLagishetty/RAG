import streamlit as st
import os
import time
from dotenv import load_dotenv

# ---- LangChain Imports (for v1.0.3) ----
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# ---- Load Environment Variables ----
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ['HF-TOKEN']=os.getenv("HF_TOKEN")

# ---- Initialize LLM ----
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)


# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using the provided context only.
    Provide a clear and concise answer.

    <context>
    {context}
    </context>

    Question: {question}
    """
)

# ---- Vector Embedding Creation ----
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# ---- Streamlit UI ----
st.set_page_config(page_title="📚 RAG with Groq & Llama3", layout="wide")
st.title("📚 RAG Document Q&A with Groq and Llama3")

user_prompt = st.text_input("🔍 Enter your question about the research papers:")

if st.button("🧠 Create Document Embeddings"):
    with st.spinner("Creating document embeddings..."):
        create_vector_embedding()
        st.success("✅ Vector database is ready!")

# ---- Query Handling ----
# ---- Query Handling ----
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first!")
    else:
        retriever = st.session_state.vectors.as_retriever()

        from langchain_core.runnables import RunnableLambda

        # ✅ FIX: Ensure retriever gets a string, not a dict
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
            }
            | prompt
            | llm
        )

        with st.spinner("Generating answer..."):
            start = time.process_time()
            response = rag_chain.invoke({"question": user_prompt})
            end_time = time.process_time() - start

            # ✅ Display the answer
            st.write("### 🧾 Answer:")
            st.write(response.content)
            st.caption(f"⏱️ Response time: {end_time:.2f} seconds")

            # ---- Document Similarity Section ----
            with st.expander("📄 Relevant Document Chunks"):
                similar_docs = retriever.get_relevant_documents(user_prompt)
                for i, doc in enumerate(similar_docs[:5]):
                    st.markdown(f"**Document {i+1}:**")
                    st.write(doc.page_content)
                    st.divider()
