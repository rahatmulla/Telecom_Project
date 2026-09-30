"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""

from langchain_core.prompts import ChatPromptTemplate           #creates prompt for llm
from langchain_core.output_parsers import StrOutputParser       #converts llm response into normal text
from langchain_core.runnables import RunnableLambda             # covenrts Python function → RunnableLambda → LangChain-compatible chain
from langchain_core.documents import Document                   #represents text+metadata
from langchain_groq import ChatGroq                             #connects to a groq hosted llm

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service. Answer questions in a direct and natural way.

Use ONLY the context below to answer the customer's question.

The context comes from three sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)
- Telecom guide (official technical and troubleshooting information)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

After each answer, add:
Sources:
- <source used>

Rules:
- Cite only sources actually used.
- Use the exact source names from the context.
- Do not invent sources.
- Put the Sources section at the very end.

Context:
{context}

"""

def _format_docs(docs: list[Document]) -> str:                      #converts retrieved docs into texts
    sections = []                                                      #list to store content of each doc
    for i, doc in enumerate(docs, start=1):                                    #goes thru each doc
        source = doc.metadata.get("source", "unknown")          #get the doc source
        sections.append(f"[{i}] {source}\n{doc.page_content}")              #add source and doc content to list
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

    def answer_question(question):
        docs = retriever.invoke(question)

        if not docs:
            return "I don't know, please call 611."

        context = _format_docs(docs)

        messages = prompt.invoke({
            "context": context,
            "question": question
        })

        response = llm.invoke(messages)

        return StrOutputParser().invoke(response)

    return RunnableLambda(answer_question)

