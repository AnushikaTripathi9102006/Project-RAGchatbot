import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Premium Appliance Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Samsung Smart Care AI",
    page_icon="🧼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for a sleek, hardware-dashboard look
st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
}

/* Background */
.stApp{
background:linear-gradient(180deg,#eef4ff,#f8fbff,#ffffff);
}

/* Hide Streamlit */
#MainMenu{
visibility:hidden;
}
footer{
visibility:hidden;
}
header{
visibility:hidden;
}

/* Hero Banner */

.hero{

background:linear-gradient(135deg,#1428A0,#246BFD);

padding:40px;

border-radius:25px;

text-align:center;

color:white;

box-shadow:0 12px 35px rgba(0,0,0,.18);

margin-bottom:25px;

}

.hero h1{

color:white!important;

font-size:46px;

margin-bottom:10px;

}

.hero p{

font-size:18px;

opacity:.9;

}

/* Glass Card */

.card{

background:rgba(255,255,255,.72);

backdrop-filter:blur(18px);

padding:30px;

border-radius:22px;

box-shadow:0 10px 30px rgba(0,0,0,.08);

border:1px solid rgba(255,255,255,.6);

}

/* Metric Cards */

.metric{

background:white;

padding:20px;

border-radius:18px;

text-align:center;

box-shadow:0 5px 18px rgba(0,0,0,.05);

transition:.3s;

}

.metric:hover{

transform:translateY(-4px);

}

/* Status */

.online{

display:inline-block;

padding:8px 18px;

border-radius:30px;

background:#dbffe7;

color:#16a34a;

font-weight:600;

}

/* Chat Bubble */

.response{

background:white;

padding:22px;

border-left:6px solid #1428A0;

border-radius:18px;

box-shadow:0 8px 18px rgba(0,0,0,.08);

font-size:17px;

line-height:1.8;

}

/* Buttons */

.stButton>button{

width:100%;

background:#1428A0;

color:white;

border:none;

padding:12px;

border-radius:12px;

font-weight:600;

transition:.3s;

}

.stButton>button:hover{

background:#246BFD;

transform:scale(1.02);

}

/* Input */

.stTextInput input{

border-radius:15px;

padding:14px;

border:2px solid #d9e5ff;

font-size:17px;

}

.stTextInput input:focus{

border:2px solid #246BFD;

}

</style>
""",unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Hardcoded Key & Environment Setup
# -----------------------------------------------------------------------------
# Using the active key from your notebook
os.environ["OPENAI_API_KEY"] = "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"

HTML_PATH = "/content/How to use the various modes of the washing machine _ Samsung LEVANT.html"

# -----------------------------------------------------------------------------
# 3. Cached RAG Setup (Ensures fast load times)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_chain():
    if not os.path.exists(HTML_PATH):
        return None
        
    # Load and split
    loader = UnstructuredHTMLLoader(file_path=HTML_PATH)
    machine_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(machine_docs)
    
    # Vectors & Model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Prompt Setup
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \nContext: {context} \nAnswer:"
    )
    
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)
    return chain

# -----------------------------------------------------------------------------
# 4. App UI Layout
# -----------------------------------------------------------------------------

# Header Block
st.markdown("""
<div class="hero">
<h1>🧺 Samsung Smart Care AI</h1>
<p>Your Intelligent Washing Machine Assistant</p>
</div>
""",unsafe_allow_html=True)

# Main UI Panel
st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
st.markdown("<span class='status-badge'>● System Online</span>", unsafe_allow_html=True)
st.subheader("What can I help you program or troubleshoot today?")

# Initialize chain with a clean spinner
with st.spinner("Syncing appliance documentation matrix..."):
    rag_chain = initialize_rag_chain()

if rag_chain is None:
    st.error(f"Could not locate the manual file at: `{HTML_PATH}`. Please verify your system path.")
else:
    # Example prompts helper pills
    st.markdown("<small style='color: #868e96;'>Suggested Queries:</small>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What is the cycle for DRUM CLEAN?", use_container_width=True):
            st.session_state.query_input = "What is the cycle for DRUM CLEAN?"
    with col2:
        if st.button("What should I do for Super echo wash?", use_container_width=True):
            st.session_state.query_input = "What should I do for Super echo wash?"

    # Chat Input text field
    query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., How do I handle an 4C error or use Outdoor Care?", 
        key="query_input",
        label_visibility="collapsed"
    )

    # Action execution
    if query:
        st.markdown("---")
        with st.spinner("Analyzing manual..."):
            try:
                answer = rag_chain.invoke(query).content
                
                # Output Block styled like a premium screen interface
                st.markdown("### 🤖 Appliance Guide Response")
                st.info(answer)
                
                # TTS Hook Visual representation (as requested in the project brief)
                st.markdown("""
                    <div class='voice-ready'>
                        <span class='voice-dot'></span>
                        <small><i>Ready for Text-to-Speech playback engine output.</i></small>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; color: #adb5bd; font-size: 0.75rem; margin-top: 5px;'>
        Samsung PoC Dashboard • Powered by LangChain & GPT-4o-Mini
    </div>
""", unsafe_allow_html=True)
