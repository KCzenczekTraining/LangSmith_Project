import nest_asyncio
from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
from typing import List
from utils import get_vector_db_retriever


load_dotenv()


# # I - LLM Runs

# inputs = [
#   {"role": "system", "content": "You are a helpful assistant."},
#   {"role": "user", "content": "I'd like to book a table for two."},
# ]

# output = {
#   "choices": [
#       {
#           "message": {
#               "role": "assistant",
#               "content": "Sure, what time would you like to book the table for?"
#           }
#       }
#   ]
# }

# # Can also use one of:
# # output = {
# #     "message": {
# #         "role": "assistant",
# #         "content": "Sure, what time would you like to book the table for?"
# #     }
# # }
# #
# # output = {
# #     "role": "assistant",
# #     "content": "Sure, what time would you like to book the table for?"
# # }
# #
# # output = ["assistant", "Sure, what time would you like to book the table for?"]


# @traceable(
#   run_type="llm", 
#   metadata={
#     "ls_provider": "openai",  # "anthropic"
#     "ls_model_name": "gpt-4o-mini",  # "claude-sonnet-4-20250514"
#   }
# )
# def chat_model(messages: list):
#   return output

# result = chat_model(inputs)

# print(result)


# # II - Handling Streaming LLM Runs

# def _reduce_chunks(chunks: list):
#     all_text = "".join([chunk["choices"][0]["message"]["content"] for chunk in chunks])
#     return {"choices": [{"message": {"content": all_text, "role": "assistant"}}]}

# @traceable(
#     run_type="llm",
#     metadata={"ls_provider": "my_provider", "ls_model_name": "my_model"},
#     reduce_chunks=_reduce_chunks
# )
# def my_streaming_chat_model(messages: list):
#     for chunk in ["Hello, " + messages[1]["content"]]:
#         yield {
#             "choices": [
#                 {
#                     "message": {
#                         "content": chunk,
#                         "role": "assistant",
#                     }
#                 }
#             ]
#         }

# result_1 = list(
#     my_streaming_chat_model(
#         [
#             {"role": "system", "content": "You are a helpful assistant. Please greet the user."},
#             {"role": "user", "content": "polly the parrot"},
#         ],
#     )
# )
# print(result_1)


# # III - Retriever Runs + Documents

# def _convert_docs(results):
#   return [
#       {
#           "page_content": r,
#           "type": "Document", # The key should be type
#           "metadata": {"foo": "bar"}
#       }
#       for r in results
#   ]

# @traceable(
#     run_type="retriever"
# )
# def retrieve_docs(query):
#   # Retriever returning hardcoded dummy documents.
#   # In production, this could be a real vector datatabase or other document index.
#   contents = ["Document contents 1", "Document contents 2", "Document contents 3"]
#   return _convert_docs(contents)

# retrieve_docs("User query")


# IV - Tool Calling

from langsmith import traceable
from openai import OpenAI
from typing import List, Optional
import json

openai_client = OpenAI()

tools = [
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["Celsius", "Fahrenheit"],
              "description": "The temperature unit to use. Infer this from the user's location."
            }
          },
          "required": ["location", "unit"]
        }
      }
    }
]
inputs = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the weather today in New York City?"},
]


@traceable(
  run_type="tool"
)
def get_current_temperature(location: str, unit: str):
    return 65 if unit == "Fahrenheit" else 17

@traceable(run_type="llm")
def call_openai(
    messages: List[dict], tools: Optional[List[dict]]
) -> str:
  return openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0,
    tools=tools
  )

@traceable(run_type="chain")
def ask_about_the_weather(inputs, tools):
  response = call_openai(inputs, tools)
  tool_call_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
  location = tool_call_args["location"]
  unit = tool_call_args["unit"]
  tool_response_message = {
    "role": "tool",
    "content": json.dumps({
        "location": location,
        "unit": unit,
        "temperature": get_current_temperature(location, unit),
    }),
    "tool_call_id": response.choices[0].message.tool_calls[0].id
  }
  inputs.append(response.choices[0].message)
  inputs.append(tool_response_message)
  output = call_openai(inputs, None)
  return output

ask_about_the_weather(inputs, tools)