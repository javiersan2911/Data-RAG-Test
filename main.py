from fastapi import FastAPI

from agents.faq_agent import mh_rag_agent_executor
from models.faq_rag_query import FAQQueryInput, FAQQueryOutput

app = FastAPI(
    title="Mental Health Counseling",
    description="Endpoints for a RAG system to answer mental health related questions",
)


@app.get("/")
def get_status():
    return {"status": "running"}


@app.post("/mh-rag-agent")
def query_mh_agent(query: FAQQueryInput) -> FAQQueryOutput:
    query_response = mh_rag_agent_executor.invoke({"input": query.text})
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response
