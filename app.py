import os
import tempfile
import streamlit as st
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Samsung Washing Machine Assistant",
    page_icon="🧺",
    layout="wide"
)

st.title("🧺 Samsung Washing Machine RAG Chatbot")
st.write("Upload the Samsung washing machine HTML manual and ask questions.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload HTML Manual",
    type=["html", "htm"]
)


# -----------------------------
# Build Vector Store
# -----------------------------
@st.cache_resource
def load_vectorstore(file_bytes, api_key):

    os.environ["OPENAI_API_KEY"] = api_key

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name

    # Read HTML
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # Extract text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0
    
    )

    try:
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    except Exception as e:
        st.error(f"Error creating vector store:\n\n{e}")
        st.stop()
    
    return vectorstore


# -----------------------------
# Main App
# -----------------------------
if uploaded_file and api_key:

    vectorstore = load_vectorstore(
        uploaded_file.getvalue(),
        api_key
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful Samsung Washing Machine Assistant.

Answer ONLY using the provided context.

If the answer is not found in the manual, say:

"I couldn't find that information in the manual."

Question:
{question}

Context:
{context}

Answer:
"""
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question...")

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        final_prompt = prompt.format(
            question=question,
            context=context
        )

        response = llm.invoke(final_prompt)

        answer = response.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.expander("Retrieved Context"):
            st.write(context)

else:
    st.info("👈 Enter your OpenAI API key and upload the HTML manual.")
