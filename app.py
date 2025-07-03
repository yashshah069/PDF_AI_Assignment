import streamlit as st
from dotenv import load_dotenv
from utils import extract_chunks_from_pdf, embed_documents
from agents_setup import create_router_agent
from agents import Runner
import asyncio
import os

load_dotenv(dotenv_path="key.env")
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF AI Agent", layout="wide")
st.title("📄 Multi-Agent PDF Assistant")
st.markdown("Upload a PDF and ask questions or request a summary!")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if (not st.session_state.pdf_processed or 
        st.session_state.current_pdf_name != uploaded_file.name):
        
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                st.session_state.messages = []
                st.session_state.pdf_processed = False
                st.session_state.retriever = None
                
                docs = extract_chunks_from_pdf(uploaded_file)
                if not docs:
                    st.error("No text could be extracted from the PDF")
                    st.stop()
                
                vectorstore = embed_documents(docs)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.session_state.pdf_processed = True
                st.session_state.current_pdf_name = uploaded_file.name
                
                st.success("PDF processed successfully! You can now ask questions.")
                
                st.session_state.messages.append({
                    "role": "system", 
                    "content": f"Document '{uploaded_file.name}' has been processed and is ready for questions."
                })
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.pdf_processed = False
                st.session_state.retriever = None
                st.stop()

if st.session_state.retriever:
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    if query := st.chat_input("Ask a question about the document or say 'summarize'"):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    router = create_router_agent()
                    context = {"retriever": st.session_state.retriever}
                    
                    result = asyncio.run(Runner.run(router, query, context=context))
                    response = result.final_output
                    
                    st.write(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. **Upload a PDF** using the file uploader
    2. **Wait for processing** - the system will chunk and embed your document
    3. **Ask questions** or request a summary
    
    ### Example queries:
    - "What is this document about?"
    - "Summarize the main points"
    - "What are the key findings?"
    - "Tell me about [specific topic]"
    """)
    
    if st.session_state.pdf_processed:
        st.success("Document ready!")
        if st.button("Clear Document"):
            st.session_state.retriever = None
            st.session_state.messages = []
            st.session_state.pdf_processed = False
            st.session_state.current_pdf_name = None
            st.rerun()
    else:
        st.info("📄 Please upload a PDF to begin")

if not st.session_state.retriever:
    st.info("Upload a PDF document to start chatting!")
else:
    st.sidebar.info(f"{len([m for m in st.session_state.messages if m['role'] != 'system'])} messages in conversation")