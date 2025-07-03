import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="key.env")
api_key = os.getenv("OPENAI_API_KEY")

def extract_chunks_from_pdf(file, chunk_size=1000, chunk_overlap=100):
    try:
        all_text = ""
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text += f"\n[Page {page_num + 1}]\n{text}\n"
        
        if not all_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        texts = text_splitter.split_text(all_text)
        
        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
                metadata={
                    "chunk_id": i,
                    "source": file.name if hasattr(file, 'name') else "uploaded_pdf",
                    "total_chunks": len(texts)
                }
            )
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def embed_documents(documents):
    try:
        if not documents:
            raise ValueError("No documents provided for embedding")

        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-ada-002"
        )

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Error creating embeddings: {str(e)}")