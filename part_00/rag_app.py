import nest_asyncio
from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
from typing import List
from utils import get_vector_db_retriever


load_dotenv()


MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o-mini"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
"""

openai_client = OpenAI()
nest_asyncio.apply()
retriever = get_vector_db_retriever()


@traceable(run_type="chain")
def retrieve_documents(question: str):
    """
    retrieve_documents
    - Returns documents fetched from a vectorstore based on the user's question
    """
    return retriever.invoke(question)


@traceable(run_type="chain")
def generate_response(question: str, documents):
    """
    generate_response
    - Calls `call_openai` to generate a model response after formatting inputs
    """
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_openai(messages)


@traceable(run_type="llm")
def call_openai(
    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0
) -> str:
    """
    call_openai
    - Returns the chat completion output from OpenAI
    """
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


@traceable(run_type="chain")
def langsmith_rag(question: str):
    """
    langsmith_rag
    - Calls `retrieve_documents` to fetch documents
    - Calls `generate_response` to generate a response based on the fetched documents
    - Returns the model response
    """
    # import requests
    # print(requests.get("http://httpbin.org/user-agent").text)
    # {
    #     "user-agent": "python-requests/2.32.4"
    # }
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content

question = "What is LangSmith used for?"
ai_answer = langsmith_rag(question
                        #   ,langsmith_extra={"metadata": {"website": "www.google.com"}}
                        )
print(ai_answer)