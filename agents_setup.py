from agents import Agent
from tools import retrieve_context, get_document_summary
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="key.env")
api_key = os.getenv("OPENAI_API_KEY")

def create_rag_agent():
    return Agent(
        name="RAG Agent",
        instructions="""You are a helpful assistant that answers questions about uploaded documents.
        Use the retrieve_context tool to find relevant information from the document and provide accurate, 
        detailed answers based on the retrieved content. If you cannot find relevant information, 
        say so clearly.""",
        model="gpt-4o-mini",
        tools=[retrieve_context]
    )

def create_summary_agent():
    return Agent(
        name="Summary Agent", 
        instructions="""You are a helpful assistant that creates comprehensive summaries of documents.
        Use the get_document_summary tool to analyze the document and provide a clear, well-structured 
        summary that captures the main points and key information.""",
        model="gpt-4o-mini",
        tools=[get_document_summary]
    )

def create_router_agent():
    return Agent(
        name="Router Agent",
        instructions="""You are a routing assistant that helps users interact with documents.

        Analyze the user's request:
        - If they ask for a summary, overview, or want to know what the document is about, use the summarize_document tool
        - For specific questions about the document content, use the answer_question tool
        
        Choose the appropriate tool based on the user's intent.""",
        model="gpt-4o-mini",
        tools=[retrieve_context, get_document_summary]  
    )