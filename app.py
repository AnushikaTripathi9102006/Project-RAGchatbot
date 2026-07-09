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
        /* General background and typography adjustments */
        .stApp {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #1428a0 !important; /* Samsung Blue */
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 600;
        }
        
        /* Dashboard Card Container */
        .dashboard-card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
        }
        
        /* Status Badges */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            background-color: #e3faf2;
            color: #0ca678;
            margin-bottom: 10px;
        }
        
        /* Voice Engine Indicator */
        .voice-ready {
            font-size: 0.85rem;
            color: #495057;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 10px;
        }
        .voice-dot {
            height: 8px;
            width: 8px;
            background-color: #3b5bdb;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.6; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.6; }
        }
    </style>
""", unsafe_allow_html=True)

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
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='color: #868e96; text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem; margin-bottom: 5px;'>SmartHome Diagnostics</p>
        <h1 style='margin-top: 0;'>SAMSUNG Care AI</h1>
        <p style='color: #495057;'>Instant interactive guidance for your Smart Washing Machine</p>
    </div>
""", unsafe_allow_html=True)

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