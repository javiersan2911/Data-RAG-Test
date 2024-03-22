from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent

from chains.faq_vector_db_chain import faq_chain

mh_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Answers",
        func=faq_chain.invoke,
        description="""Useful to answer mental health
        related questions.
        """,
    )]

chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
)

mh_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=mh_agent_prompt,
    tools=tools,
)

mh_rag_agent_executor = AgentExecutor(
    agent=mh_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
