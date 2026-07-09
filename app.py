import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# -----------------------------------------------------------------------------
# 1. Page Configuration & Premium Minimalist CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Samsung Smart Care AI",
    page_icon="🧼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Deep clean, high-end appliance aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f3f7fa 0%, #ffffff 100%);
}

/* Hide Default Streamlit Elements */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Premium Hero Header */
.hero-container {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 35px 25px;
    border-radius: 24px;
    text-align: center;
    color: white;
    box-shadow: 0 20px 40px rgba(15, 32, 39, 0.15);
    margin-bottom: 30px;
}

.hero-container h1 {
    color: #ffffff !important;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}

.hero-container p {
    font-size: 16px;
    font-weight: 300;
    color: #e0e6ed;
    margin: 0;
}

/* Dashboard Core Card */
.dashboard-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    border: 1px solid #eef2f5;
    margin-bottom: 25px;
}

/* Animated Glowing Status Badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 100px;
    background: #e6f9ed;
    color: #10b981;
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 20px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse 1.6s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Custom Text Input Styling Override */
.stTextInput input {
    border-radius: 14px !important;
    padding: 14px 18px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 16px !important;
    background-color: #f8fafc !important;
    transition: all 0.2s ease-in-out;
}

.stTextInput input:focus {
    border-color: #203a43 !important;
    background-color: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(32, 58, 67, 0.1) !important;
}

/* Custom Button Styling */
.stButton>button {
    width: 100%;
    background-color: #ffffff;
    color: #334155;
    border: 1px solid #e2e8f0;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.2s;
}

.stButton>button:hover {
    background-color: #f8fafc;
    border-color: #cbd5e1;
    color: #0f2027;
}

/* Response Section Styling */
.response-header {
    font-size: 18px;
    font-weight: 600;
    color: #0f2027;
    margin-top: 25px;
    margin-bottom: 12px;
}

/* TTS Hook Indicator */
.voice-ready {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 15px;
    padding: 10px 14px;
    background: #f1f5f9;
    border-radius: 10px;
    color: #475569;
}

.voice-dot {
    width: 6px;
    height: 6px;
    background: #64748b;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Key & Environment Setup
# -----------------------------------------------------------------------------
# Safely fall back to your active key string if not configured in the global env
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY", 
    "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"
)

HTML_PATH = "/content/How to use the various modes of the washing machine _ Samsung LEVANT.html"

# -----------------------------------------------------------------------------
# 3. Cached RAG Setup
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_rag_chain():
    if not os.path.exists(HTML_PATH):
        return None
        
    loader = UnstructuredHTMLLoader(file_path=HTML_PATH)
    machine_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(machine_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \nContext: {context} \nAnswer:"
    )
    
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)
    return chain

# Helper function to assign dynamic suggestions without breaking input lifecycle
def set_suggestion(text):
    st.session_state.query_input = text

# -----------------------------------------------------------------------------
# 4. App UI Layout
# -----------------------------------------------------------------------------

# Premium Hero Banner
st.markdown("""
<div class="hero-container">
    <h1>🧺 Samsung Smart Care AI</h1>
    <p>Premium Appliance Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Main Structural Dashboard Card Block
st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
st.markdown("<div class='status-badge'><span class='status-dot'></span>System Connected</div>", unsafe_allow_html=True)

st.markdown("<h3 style='margin-top:0; margin-bottom: 20px; font-size:20px; color:#1e293b;'>What can I help you program or troubleshoot today?</h3>", unsafe_allow_html=True)

# Initialize chain with a clean inline layout element
with st.spinner("Syncing appliance documentation matrix..."):
    rag_chain = initialize_rag_chain()

if rag_chain is None:
    st.error(f"Could not locate the manual file at: `{HTML_PATH}`. Please verify your system path.")
else:
    # Pill Helpers with interactive callback assignments
    st.markdown("<p style='color: #64748b; font-size:13px; margin-bottom:8px; font-weight:500;'>Suggested Enquiries:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.button("What is the cycle for DRUM CLEAN?", use_container_width=True, on_click=set_suggestion, args=("What is the cycle for DRUM CLEAN?",))
    with col2:
        st.button("What should I do for Super echo wash?", use_container_width=True, on_click=set_suggestion, args=("What should I do for Super echo wash?",))

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # Chat Input text field
    query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., How do I handle a 4C error or use Outdoor Care?", 
        key="query_input",
        label_visibility="collapsed"
    )

    # Process Query Action
    if query:
        st.markdown("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 25px 0;'>", unsafe_allow_html=True)
        with st.spinner("Analyzing manual structure..."):
            try:
                answer = rag_chain.invoke(query).content
                
                # Output Block Styled like a premium OLED interface panel
                st.markdown("<div class='response-header'>🤖 Appliance Guide Response</div>", unsafe_allow_html=True)
                st.info(answer)
                
                # TTS Hook Visual representation
                st.markdown("""
                    <div class='voice-ready'>
                        <span class='voice-dot'></span>
                        <small style='font-size:12px; font-weight:500;'>Ready for Text-to-Speech playback engine output.</small>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Bottom Footer
st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 15px;'>
        Samsung PoC Control Unit • Powered by LangChain & GPT-4o-Mini
    </div>
""", unsafe_allow_html=True)
