from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

FAQ_CHROMA_PATH = "../data/vector"

faq_template_str = """You are an assistant for question-answering tasks. Use the following pieces of retrieved 
context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences 
maximum and keep the answer concise. Question: {question} Context: {context} Answer:"""

faq_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=faq_template_str
    )
)

faq_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [faq_system_prompt, faq_human_prompt]

faq_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

output_parser = StrOutputParser()

faqs_retriever = Chroma(
    embedding_function=OpenAIEmbeddings(), persist_directory=FAQ_CHROMA_PATH
).as_retriever(k=10)

faq_chain = (
        {"context": faqs_retriever, "question": itemgetter("question")}
        | faq_prompt_template
        | chat_model
        | StrOutputParser()
)
