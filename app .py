import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain.tools import tool
from langchain.agents import create_agent

import faiss


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

CHAT_MODEL = os.environ.get(
    "GOOGLE_CHAT_MODEL",
    "gemini-2.5-flash"
)

EMBEDDING_MODEL = os.environ.get(
    "GOOGLE_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE_TEXT = """
The Internet is a global system of interconnected computer
networks that uses the Internet protocol suite (TCP/IP) to
communicate between networks and devices.

It is a network of networks that consists of private, public,
academic, business, and government networks of local to global
scope, linked by electronic, wireless, and optical networking
technologies.

The Internet carries a vast range of information resources and
services, such as the World Wide Web, electronic mail,
telephony, and file sharing.

The origins of the Internet date back to packet switching
research commissioned by the United States Department of
Defense in the 1960s.

The ARPANET initially served as a backbone for interconnection
of academic and research networks.

The commercialization of the Internet in the mid-1990s marked
a turning point in its expansion.

Today, the Internet is a pervasive global information medium.

It supports cloud computing, video conferencing, online gaming,
social media, education, commerce, healthcare, and communication.

The Internet also presents challenges related to privacy,
security, and misinformation.
"""


# ============================================================
# PROMPTS
# ============================================================

RAG_SYSTEM_INSTRUCTIONS = """
You are a helpful assistant.

Use ONLY the retrieved context to answer the question.

If the context does not contain the answer, say:
"I don't know based on the provided context."

Do not invent information.
"""

AGENT_SYSTEM_PROMPT = """
You are a helpful assistant.

You have access to a tool that retrieves information from
the Internet knowledge base.

Use ONLY the retrieved context to answer questions.

If the context does not contain the answer, say:
"I don't know based on the provided context."

Do not invent information.
"""


# ============================================================
# GLOBAL VARIABLES
# ============================================================

llm = None
vector_store = None
retriever = None
internet_agent = None


# ============================================================
# INITIALIZE AI
# ============================================================

def initialize_ai():

    global llm
    global vector_store
    global retriever
    global internet_agent

    if llm is not None and vector_store is not None:
        return

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. "
            "Add GOOGLE_API_KEY in Render Environment Variables."
        )

    print("Initializing Google AI...")

    # --------------------------------------------------------
    # Google Gemini model
    # --------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    documents = [
        Document(
            page_content=KNOWLEDGE_BASE_TEXT,
            metadata={
                "source": "Internet Knowledge Base"
            }
        )
    ]

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # --------------------------------------------------------
    # Google embeddings
    # --------------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    # --------------------------------------------------------
    # Get embedding dimension
    # --------------------------------------------------------

    test_embedding = embeddings.embed_query(
        "hello world"
    )

    embedding_dimension = len(test_embedding)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = faiss.IndexFlatL2(
        embedding_dimension
    )

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(chunks)

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 2
        }
    )

    # --------------------------------------------------------
    # Agent tool
    # --------------------------------------------------------

    @tool
    def retrieve_internet_context(query: str) -> str:
        """
        Retrieve information from the Internet knowledge base.
        """

        docs = vector_store.similarity_search(
            query,
            k=2
        )

        return "\n\n".join(
            f"Source: {doc.metadata}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    internet_agent = create_agent(
        llm,
        [retrieve_internet_context],
        system_prompt=AGENT_SYSTEM_PROMPT
    )

    print("Google AI initialized successfully!")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Google AI RAG Server",
    version="1.0.0",
    description="Google Gemini RAG and Agentic RAG API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Google AI RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "google_api_key_configured": bool(GOOGLE_API_KEY)
    }


# ============================================================
# RAG ENDPOINT
# ============================================================

@app.post("/rag")
async def rag_api(data: dict):

    initialize_ai()

    question = data.get("question", "").strip()

    if not question:

        return {
            "error": "Please provide a question."
        }

    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"Source: {doc.metadata}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

    prompt = ChatPromptTemplate.from_template(
        RAG_SYSTEM_INSTRUCTIONS
        + """

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "question": question,
        "answer": answer
    }


# ============================================================
# AGENT ENDPOINT
# ============================================================

@app.post("/agent")
async def agent_api(data: dict):

    initialize_ai()

    question = data.get("question", "").strip()

    if not question:

        return {
            "error": "Please provide a question."
        }

    result = internet_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    final_message = result["messages"][-1]

    content = final_message.content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    parts.append(
                        block.get("text", "")
                    )

        answer = "\n".join(parts)

    else:

        answer = str(content)

    return {
        "question": question,
        "answer": answer
    }


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
