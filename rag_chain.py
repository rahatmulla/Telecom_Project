"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""

from langchain_core.prompts import ChatPromptTemplate           #creates prompt for llm
from langchain_core.output_parsers import StrOutputParser       #converts llm response into normal text
from langchain_core.runnables import RunnablePassthrough        #passes the input withou changing it
from langchain_core.documents import Document                   #represents text+metadata
from langchain_groq import ChatGroq                             #connects to a groq hosted llm

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from two sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:                      #converts retrieved docs into texts
    sections = []                                                      #list to store content if each doc
    for doc in docs:                                                    #goes thru each doc
        source = doc.metadata.get("source", "unknown").upper()          #get the doc source
        sections.append(f"[{source}]\n{doc.page_content}")              #add source and doc content to list
    return "\n\n---\n\n".join(sections)                                 #combine all docs into one piece of text


#creating the complete rag chain
def build_chain():                                              
    retriever = build_retriever()                                       

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),                                         #gives llm its main instructions
        ("human", "{question}"),                                           #give user question
    ])

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        max_tokens=499,                                            #no max token limit
        reasoning_format="parsed",                                  #structured output format
        timeout=None,
        max_retries=2,                                              #if request fails, retry 2 timesq
    )

    #builds the complete pipeline
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}        #give the LLM both the relevant information and the user's question.
        | prompt
        | llm
        | StrOutputParser()                                                         #turns the response into normal text
    )
    return chain
