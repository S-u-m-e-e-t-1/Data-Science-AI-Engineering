from langchain_ibm import ChatWatsonx
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import (
    PARAMETERS,
    CREDENTIALS,
    LLAMA_MODEL_ID,
    GRANITE_MODEL_ID,
    MISTRAL_MODEL_ID,
)


# -----------------------------
# Expected JSON Output
# -----------------------------
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(
        description="Sentiment score from 0 (negative) to 100 (positive)"
    )
    response: str = Field(description="Suggested response to the user")


json_parser = JsonOutputParser(pydantic_object=AIResponse)


# -----------------------------
# Model Initialization
# -----------------------------
def initialize_model(model_id):
    return ChatWatsonx(
        model_id=model_id,
        url=CREDENTIALS["url"],
        project_id=CREDENTIALS["project_id"],
        params=PARAMETERS,
    )


llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)


# -----------------------------
# Prompt Templates
# -----------------------------
llama_template = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}

{format_prompt}

<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["system_prompt", "format_prompt", "user_prompt"],
)

granite_template = PromptTemplate(
    template="""System:
{system_prompt}

{format_prompt}

Human:
{user_prompt}

AI:""",
    input_variables=["system_prompt", "format_prompt", "user_prompt"],
)

mistral_template = PromptTemplate(
    template="""<s>[INST]
{system_prompt}

{format_prompt}

{user_prompt}
[/INST]""",
    input_variables=["system_prompt", "format_prompt", "user_prompt"],
)


# -----------------------------
# Generic Response Function
# -----------------------------
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser

    return chain.invoke(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "format_prompt": json_parser.get_format_instructions(),
        }
    )


# -----------------------------
# Model-specific Functions
# -----------------------------
def llama_response(system_prompt, user_prompt):
    return get_ai_response(
        llama_llm,
        llama_template,
        system_prompt,
        user_prompt,
    )


def granite_response(system_prompt, user_prompt):
    return get_ai_response(
        granite_llm,
        granite_template,
        system_prompt,
        user_prompt,
    )


def mistral_response(system_prompt, user_prompt):
    return get_ai_response(
        mistral_llm,
        mistral_template,
        system_prompt,
        user_prompt,
    )


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    result = granite_response(
        "You are a helpful assistant.",
        "I failed my exam and I'm feeling disappointed."
    )

    print(result)