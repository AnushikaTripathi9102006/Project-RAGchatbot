import streamlit as st
import tempfile
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Samsung AI Assistant",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.stChatMessage{
    border-radius:18px;
    padding:12px;
}

.stButton>button{
    width:100%;
    border-radius:12px;
}

.stTextInput>div>div>input{
    border-radius:12px;
}

h1{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🧺 Samsung Washing Machine AI Assistant")

st.caption(
    "Ask anything about your Samsung Washing Machine manual using AI + Retrieval Augmented Generation (RAG)."
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("⚙ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Paste your OpenAI API Key."
    )

    uploaded_file = st.file_uploader(
        "Upload HTML Manual",
        type=["html", "htm"]
    )

    st.divider()

    st.markdown(
        """
### Features

✅ GPT-4o-mini

✅ Chroma Vector DB

✅ RAG Search

✅ Chat History

✅ BeautifulSoup HTML Parser

✅ Streamlit
"""
    )

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# BUILD VECTOR DATABASE
# -------------------------------------------------

@st.cache_resource(show_spinner=False)
def create_vectorstore(file_bytes, api_key):
    """
    Creates a Chroma Vector Database from the uploaded HTML.
    """

    try:

        # Save uploaded HTML temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp:

            temp.write(file_bytes)
            temp_path = temp.name

        # Read HTML
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n")

        # Clean blank lines
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        if len(text) < 100:
            raise ValueError(
                "The uploaded HTML contains very little readable text."
            )

        docs = [Document(page_content=text)]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        return vectorstore

    except Exception as e:
        raise RuntimeError(str(e))


# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------

vectorstore = None
retriever = None
llm = None

if uploaded_file and api_key:

    with st.spinner("📚 Processing washing machine manual..."):

        try:

            vectorstore = create_vectorstore(
                uploaded_file.getvalue(),
                api_key
            )

            retriever = vectorstore.as_retriever(
                search_kwargs={"k":3}
            )

            llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini",
                temperature=0
            )

            st.success("✅ Manual loaded successfully!")

        except Exception as e:

            error = str(e).lower()

            if "authentication" in error:
                st.error("❌ Invalid OpenAI API Key.")

            elif "quota" in error:
                st.error("❌ Your OpenAI account has no remaining credits.")

            elif "rate" in error:
                st.error("❌ Rate limit exceeded. Try again later.")

            else:
                st.error(f"❌ {e}")

            st.stop()
# -------------------------------------------------
# PROMPT TEMPLATE
# -------------------------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are Samsung's official Washing Machine AI Assistant.

Answer ONLY using the provided context.

Rules:
- Give short and clear answers.
- If the answer is not in the manual, say:
  "I couldn't find that information in the manual."
- Do not make up information.

Question:
{question}

Context:
{context}

Answer:
"""
)

# -------------------------------------------------
# DISPLAY PREVIOUS CHAT
# -------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------

if vectorstore:

    question = st.chat_input(
        "Ask something about your washing machine..."
    )

    if question:

        # Store User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Assistant
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    docs = retriever.invoke(question)

                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                    final_prompt = prompt.format(
                        question=question,
                        context=context
                    )

                    response = llm.invoke(final_prompt)

                    answer = response.content

                except Exception as e:

                    answer = f"❌ Error: {e}"

                st.markdown(answer)

        # Save Assistant Message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Show Retrieved Chunks
        with st.expander("📖 Retrieved Context"):

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### Chunk {i}")

                st.write(doc.page_content)

                st.divider()

else:

    st.info("👈 Upload an HTML manual and enter your OpenAI API key.")

# -------------------------------------------------
# SIDEBAR TOOLS
# -------------------------------------------------

with st.sidebar:

    st.divider()

    st.subheader("📊 Session")

    if vectorstore:
        st.success("🟢 Manual Loaded")
    else:
        st.warning("🔴 No Manual Loaded")

    st.write("**Model:** GPT-4o-mini")
    st.write("**Embeddings:** text-embedding-3-small")
    st.write("**Retriever:** ChromaDB")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# -------------------------------------------------
# SUGGESTED QUESTIONS
# -------------------------------------------------

if vectorstore:

    st.divider()

    st.subheader("💡 Try asking")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🧺 What is Drum Clean?"):

            st.session_state.messages.append(
                {
                    "role":"user",
                    "content":"What is Drum Clean?"
                }
            )

            st.rerun()

        if st.button("🌱 What is Super Eco Wash?"):

            st.session_state.messages.append(
                {
                    "role":"user",
                    "content":"What is Super Eco Wash?"
                }
            )

            st.rerun()

    with col2:

        if st.button("⚠️ What do warning symbols mean?"):

            st.session_state.messages.append(
                {
                    "role":"user",
                    "content":"Explain the warning symbols."
                }
            )

            st.rerun()

        if st.button("🧼 How often should I clean the drum?"):

            st.session_state.messages.append(
                {
                    "role":"user",
                    "content":"How often should I clean the drum?"
                }
            )

            st.rerun()

# -------------------------------------------------
# DOWNLOAD CHAT
# -------------------------------------------------

if st.session_state.messages:

    transcript = ""

    for msg in st.session_state.messages:

        transcript += f"{msg['role'].upper()}\n"

        transcript += msg["content"]

        transcript += "\n\n"

    st.download_button(

        label="📥 Download Chat",

        data=transcript,

        file_name="chat_history.txt",

        mime="text/plain"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.markdown(
"""
<div style='text-align:center;color:gray;font-size:14px;'>

🧺 Samsung Washing Machine AI Assistant

Built using <b>Streamlit</b> • <b>LangChain</b> •
<b>OpenAI</b> • <b>ChromaDB</b>

</div>
""",
unsafe_allow_html=True
)
