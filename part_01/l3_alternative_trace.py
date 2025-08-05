import nest_asyncio
from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
from typing import List
from utils import get_vector_db_retriever


load_dotenv()


# # I - LangChain and LangGraph

# import nest_asyncio
# import operator
# from langchain.schema import Document
# from langchain_core.messages import HumanMessage, AnyMessage, get_buffer_string
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END
# from typing import List
# from typing_extensions import TypedDict, Annotated
# from utils import get_vector_db_retriever, RAG_PROMPT

# nest_asyncio.apply()

# retriever = get_vector_db_retriever()
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# # Define Graph state
# class GraphState(TypedDict):
#     question: str
#     messages: Annotated[List[AnyMessage], operator.add]
#     documents: List[Document]

# # Define Nodes
# def retrieve_documents(state: GraphState):
#     messages = state.get("messages", [])
#     question = state["question"]
#     documents = retriever.invoke(f"{get_buffer_string(messages)} {question}")
#     return {"documents": documents}

# def generate_response(state: GraphState):
#     question = state["question"]
#     messages = state["messages"]
#     documents = state["documents"]
#     formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    
#     rag_prompt_formatted = RAG_PROMPT.format(context=formatted_docs, conversation=messages, question=question)
#     generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
#     return {"documents": documents, "messages": [HumanMessage(question), generation]}

# # Define Graph
# graph_builder = StateGraph(GraphState)
# graph_builder.add_node("retrieve_documents", retrieve_documents)
# graph_builder.add_node("generate_response", generate_response)
# graph_builder.add_edge(START, "retrieve_documents")
# graph_builder.add_edge("retrieve_documents", "generate_response")
# graph_builder.add_edge("generate_response", END)

# simple_rag_graph = graph_builder.compile()

# # Save the output as an image file in the part_01 folder
# with open("part_01/simple_rag_graph.png", "wb") as f:
#     f.write(simple_rag_graph.get_graph().draw_mermaid_png())

# question = "How do I set up tracing if I'm using LangChain?"
# simple_rag_graph.invoke({"question": question}, config={"metadata": {"foo": "bar"}})


# # II - Tracing Context Manager

# from langsmith import traceable, trace
# from openai import OpenAI
# from typing import List
# import nest_asyncio
# from utils import get_vector_db_retriever


# MODEL_PROVIDER = "openai"
# MODEL_NAME = "gpt-4o-mini"
# APP_VERSION = 1.0
# RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the latest question in the conversation. 
# If you don't know the answer, just say that you don't know. 
# Use three sentences maximum and keep the answer concise.
# """

# openai_client = OpenAI()
# nest_asyncio.apply()
# retriever = get_vector_db_retriever()


# @traceable
# def retrieve_documents(question: str):
#     """
#     retrieve_documents
#     - Returns documents fetched from a vectorstore based on the user's question
#     """
#     documents = retriever.invoke(question)
#     return documents


# #  @traceableis is removed and with trace() used instead
# def generate_response(question: str, documents):
#     """
#     generate_response
#     - Calls `call_openai` to generate a model response after formatting inputs
#     """
#     # NOTE: Our documents came in as a list of objects, but we just want to log a string
#     formatted_docs = "\n\n".join(doc.page_content for doc in documents)


#     with trace(
#         name="Generate Response",
#         run_type="chain", 
#         inputs={"question": question, "formatted_docs": formatted_docs},
#         metadata={"foo": "bar"},
#     ) as ls_trace:
#         messages = [
#             {
#                 "role": "system",
#                 "content": RAG_SYSTEM_PROMPT
#             },
#             {
#                 "role": "user",
#                 "content": f"Context: {formatted_docs} \n\n Question: {question}"
#             }
#         ]
#         response = call_openai(messages)
    
#         # End your trace and write outputs to LangSmith
#         ls_trace.end(outputs={"output": response})
#     return response


# @traceable
# def call_openai(
#     messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0
# ) -> str:
#     """
#     call_openai
#     - Returns the chat completion output from OpenAI
#     """
#     response = openai_client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#     )
#     return response


# @traceable
# def langsmith_rag(question: str):
#     """
#     langsmith_rag
#     - Calls `retrieve_documents` to fetch documents
#     - Calls `generate_response` to generate a response based on the fetched documents
#     - Returns the model response
#     """
#     documents = retrieve_documents(question)
#     response = generate_response(question, documents)

#     result = response.choices[0].message.content
#     return result

# question = "How do I trace with tracing context?"
# ai_answer = langsmith_rag(question)
# print(ai_answer)


# III - wrap_openai

from langsmith.wrappers import wrap_openai
import openai
from typing import List
import nest_asyncio
from utils import get_vector_db_retriever


MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o-mini"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
"""

# Wrap the OpenAI Client
openai_client = wrap_openai(openai.Client())

nest_asyncio.apply()
retriever = get_vector_db_retriever()

@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    # or msg_1
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
    # TODO: We don't need to use @traceable on def call_openai a nested function call anymore,
    # wrap_openai takes care of this for us
    return openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        langsmith_extra={"metadata": {"foo": "bar"}}
    )

@traceable(run_type="chain")
def langsmith_rag_with_wrap_openai(question: str):
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content

question = "How do I trace with wrap_openai?"
ai_answer = langsmith_rag_with_wrap_openai(question)
print(ai_answer)

# msg_1
messages_1 = [
    {
        "role": "user",
        "content": "What color is the sky?"
    }
]


# # IV - RunTree (for later to walk through)

# import os
# os.environ["LANGSMITH_TRACING"] = "false"

# from langsmith import utils
# utils.tracing_is_enabled() # This should return false

# from langsmith import RunTree
# from openai import OpenAI
# from typing import List
# import nest_asyncio
# from utils import get_vector_db_retriever

# openai_client = OpenAI()
# nest_asyncio.apply()
# retriever = get_vector_db_retriever()

# def retrieve_documents(parent_run: RunTree, question: str):
#     # Create a child run
#     child_run = parent_run.create_child(
#         name="Retrieve Documents",
#         run_type="retriever",
#         inputs={"question": question},
#     )
#     documents = retriever.invoke(question)
#     # Post the output of our child run
#     child_run.end(outputs={"documents": documents})
#     child_run.post()
#     return documents

# def generate_response(parent_run: RunTree, question: str, documents):
#     formatted_docs = "\n\n".join(doc.page_content for doc in documents)
#     rag_system_prompt = """You are an assistant for question-answering tasks. 
#     Use the following pieces of retrieved context to answer the latest question in the conversation. 
#     If you don't know the answer, just say that you don't know. 
#     Use three sentences maximum and keep the answer concise.
#     """
#     # Create a child run
#     child_run = parent_run.create_child(
#         name="Generate Response",
#         run_type="chain",
#         inputs={"question": question, "documents": documents},
#     )
#     messages = [
#         {
#             "role": "system",
#             "content": rag_system_prompt
#         },
#         {
#             "role": "user",
#             "content": f"Context: {formatted_docs} \n\n Question: {question}"
#         }
#     ]
#     openai_response = call_openai(child_run, messages)
#     # Post the output of our child run
#     child_run.end(outputs={"openai_response": openai_response})
#     child_run.post()
#     return openai_response

# def call_openai(
#     parent_run: RunTree, messages: List[dict], model: str = "gpt-4o-mini", temperature: float = 0.0
# ) -> str:
#     # Create a child run
#     child_run = parent_run.create_child(
#         name="OpenAI Call",
#         run_type="llm",
#         inputs={"messages": messages},
#     )
#     openai_response = openai_client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#     )
#     # Post the output of our child run
#     child_run.end(outputs={"openai_response": openai_response})
#     child_run.post()
#     return openai_response

# def langsmith_rag(question: str):
#     # Create a root RunTree
#     root_run_tree = RunTree(
#         name="Chat Pipeline",
#         run_type="chain",
#         inputs={"question": question}
#     )

#     # Pass our RunTree into the nested function calls
#     documents = retrieve_documents(root_run_tree, question)
#     response = generate_response(root_run_tree, question, documents)
#     output = response.choices[0].message.content

#     # Post our final output
#     root_run_tree.end(outputs={"generation": output})
#     root_run_tree.post()
#     return output
    
# question = "How can I trace with RunTree?"
# ai_answer = langsmith_rag(question)
# print(ai_answer)
