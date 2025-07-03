from agents import function_tool, RunContextWrapper
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI  
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="key.env")
api_key = os.getenv("OPENAI_API_KEY")

@function_tool
async def retrieve_context(context: RunContextWrapper, query: str) -> str:
    try:
        if not hasattr(context, 'context') or context.context is None:
            return "No document context available. Please upload a PDF first."
            
        retriever = context.context.get("retriever")
        if not retriever:
            return "No document has been uploaded yet. Please upload a PDF first."

        llm = OpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-instruct"  
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        
        result = qa_chain.run(query)
        return result
        
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

@function_tool
async def get_document_summary(context: RunContextWrapper, input_text: str = "") -> str:
    try:
        if not hasattr(context, 'context') or context.context is None:
            return "No document context available. Please upload a PDF first."
            
        retriever = context.context.get("retriever")
        if not retriever:
            return "No document has been uploaded yet. Please upload a PDF first."

        docs = retriever.get_relevant_documents("summary overview main points key information", k=8)
        
        if not docs:
            return "No content found in the document."
        
        content = "\n\n".join([doc.page_content for doc in docs])
        
        if len(content) > 8000:
            content = content[:8000] + "..."

        prompt_template = """
        Please provide a comprehensive summary of the following document content. 
        Focus on the main topics, key points, and important information:

        {context}

        Summary:
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        llm = OpenAI(
            temperature=0.3,
            openai_api_key=api_key,
            model_name="gpt-4o-mini"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(context=content)
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"