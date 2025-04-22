from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool, BaseTool
from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.output_parsers.enum import EnumOutputParser
from langchain_openai import ChatOpenAI
from langfuse.decorators import observe
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, List


class ContentOutputParser(BaseOutputParser):

    def parse(self, text: str) -> str:
        return text


def make_chain(prompt_template: str, output_model: Any) -> Runnable:
    """
    Create a simple chain runnable.
    """
    if isinstance(output_model, BaseModel):
        parser = PydanticOutputParser(pydantic_object=output_model)
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    else:
        parser = ContentOutputParser()
        partial_variables = {}
    prompt = PromptTemplate.from_template(template=prompt_template, partial_variables=partial_variables)
    model = ChatOpenAI(model='o4-mini', temperature=0)
    return prompt | model | parser


def make_agent(tools: List[BaseTool], prompt_template: str, output_model: Any) -> Runnable:
    """
    Create an agent runnable.
    """
    if isinstance(output_model, BaseModel):
        parser = PydanticOutputParser(pydantic_object=output_model)
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    else:
        parser = ContentOutputParser()
        partial_variables = {}
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful AI assistant"),
            ("human", prompt_template),
            ("placeholder", "{agent_scratchpad}")
        ]
    ).partial(**partial_variables)
    model = ChatOpenAI(model='04-mini', temperature=0)
    agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor | RunnableLambda(lambda x: x["output"]) | parser
