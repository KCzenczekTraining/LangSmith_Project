from dotenv import load_dotenv
from langsmith import Client
from openai import OpenAI
from typing import List
from app import langsmith_rag
from langsmith.schemas import Example, Run
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

client = OpenAI()


# def correct_label(inputs: dict, reference_outputs: dict, outputs: dict) -> dict:
#   score = outputs.get("output") == reference_outputs.get("label")
#   return {"score": int(score), "key": "correct_label"}


class Similarity_Score(BaseModel):
    similarity_score: int = Field(description="Semantic similarity score between 1 and 10, where 1 means unrelated and 10 means identical.")

# This is our evaluator
def compare_semantic_similarity(inputs: dict, reference_outputs: dict, outputs: dict):
    input_question = inputs["question"]
    reference_response = reference_outputs["output"]
    run_response = outputs["output"]
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {   
                "role": "system",
                "content": (
                    "You are a semantic similarity evaluator. Compare the meanings of two responses to a question, "
                    "Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. "
                    "Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."
                ),
            },
            {"role": "user", "content": f"Question: {input_question}\n Reference Response: {reference_response}\n Run Response: {run_response}"}
        ],
        response_format=Similarity_Score,
    )

    sim_score = completion.choices[0].message.parsed
    print(sim_score)
    return {"score": sim_score.similarity_score, "key": "similarity"}

# From Dataset Example
inputs = {
  "question": "Is LangSmith natively integrated with LangChain?"
}
reference_outputs = {
  "output": "Yes, LangSmith is natively integrated with LangChain, as well as LangGraph."
}

# From Run
outputs = {
  "output": "No, LangSmith is NOT integrated with LangChain."
}

simil_score = compare_semantic_similarity(inputs, reference_outputs, outputs)
print(f"Semantic similarity score: {simil_score}")


### Anohter way ### 
# from langsmith.schemas import Run, Example

# def compare_semantic_similarity_v2(root_run: Run, example: Example):
#     input_question = example["inputs"]["question"]
#     reference_response = example["outputs"]["output"]
#     run_response = root_run["outputs"]["output"]
    
#     completion = client.beta.chat.completions.parse(
#         model="gpt-4o",
#         messages=[
#             {   
#                 "role": "system",
#                 "content": (
#                     "You are a semantic similarity evaluator. Compare the meanings of two responses to a question, "
#                     "Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. "
#                     "Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."
#                 ),
#             },
#             {"role": "user", "content": f"Question: {input_question}\n Reference Response: {reference_response}\n Run Response: {run_response}"}
#         ],
#         response_format=Similarity_Score,
#     )

#     similarity_score = completion.choices[0].message.parsed
#     return {"score": similarity_score.similarity_score, "key": "similarity"}

# sample_run = {
#   "name": "Sample Run",
#   "inputs": {
#     "question": "Is LangSmith natively integrated with LangChain?"
#   },
#   "outputs": {
#     "output": "No, LangSmith is NOT integrated with LangChain."
#   },
#   "is_root": True,
#   "status": "success",
#   "extra": {
#     "metadata": {
#       "key": "value"
#     }
#   }
# }

# sample_example = {
#   "inputs": {
#     "question": "Is LangSmith natively integrated with LangChain?"
#   },
#   "outputs": {
#     "output": "Yes, LangSmith is natively integrated with LangChain, as well as LangGraph."
#   },
#   "metadata": {
#     "dataset_split": [
#       "AI generated",
#       "base"
#     ]
#   }
# }

# similarity_score = compare_semantic_similarity_v2(sample_run, sample_example)
# print(f"Semantic similarity score: {similarity_score}")