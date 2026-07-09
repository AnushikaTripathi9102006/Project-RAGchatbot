import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Samsung Washing Machine Assistant",
    page_icon="🧺",
    layout="wide"
)

st.title("🧺 Samsung Washing Machine RAG Chatbot")
st.write("Ask questions about the Samsung Washing Machine Manual.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Samsung Manual (HTML)",
    type=["html", "htm"]
)

# -----------------------------
# Load Vector Store
# -----------------------------
@st.cache_resource
def load_vectorstore(file_path, api_key):

    os.environ["OPENAI_API_KEY"] = api_key

    loader = UnstructuredHTMLLoader(file_path=file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore


if uploaded_file and api_key:

    temp_path = "manual.html"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    vectorstore = load_vectorstore(temp_path, api_key)

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert Samsung Washing Machine assistant.

Answer ONLY from the provided context.

If the answer is not available in the manual,
reply:

"I couldn't find that information in the manual."

Question:
{question}

Context:
{context}

Answer:
"""
    )

    st.success("✅ Manual Loaded Successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    question = st.chat_input("Ask something about the washing machine...")

    if question:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        formatted_prompt = prompt.format(
            question=question,
            context=context
        )

        response = llm.invoke(formatted_prompt)

        answer = response.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        with st.expander("Retrieved Context"):
            st.write(context)

else:
    st.info("Upload the HTML manual and enter your OpenAI API key.")
st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 15px;'>
        Samsung PoC Control Unit • Powered by LangChain & GPT-4o-Mini
    </div>
""", unsafe_allow_html=True)
