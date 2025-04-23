from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Any, List

MODEL = 'gpt-3.5-turbo'


class ContentOutputParser(BaseOutputParser):

    def parse(self, text: str) -> str:
        return text


def make_chain(prompt_template: str, output_model: Any) -> Runnable:
    """
    Create a simple chain runnable.
    """
    if issubclass(output_model, BaseModel):
        parser = PydanticOutputParser(pydantic_object=output_model)
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    else:
        parser = ContentOutputParser()
        partial_variables = {}
    chain_prompt = PromptTemplate.from_template(template=prompt_template, partial_variables=partial_variables)
    model = ChatOpenAI(model=MODEL, temperature=0)
    return chain_prompt | model | parser


def make_agent(tools: List[BaseTool], prompt_template: str, output_model: Any) -> Runnable:
    """
    Create an agent runnable.
    """
    if issubclass(output_model, BaseModel):
        parser = PydanticOutputParser(pydantic_object=output_model)
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    else:
        parser = ContentOutputParser()
        partial_variables = {}
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful AI assistant"),
            ("human", prompt_template),
            ("placeholder", "{agent_scratchpad}")
        ]
    ).partial(**partial_variables)
    model = ChatOpenAI(model=MODEL, temperature=0)
    agent = create_tool_calling_agent(llm=model, tools=tools, prompt=agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
    return agent_executor | RunnableLambda(lambda x: x["output"]) | parser
