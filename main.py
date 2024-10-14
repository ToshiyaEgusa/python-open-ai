import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


def initialize_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings()

    vector_store_path = "./resources.db"

    unlisted_df = pd.read_csv("./resources/unlisted.csv")
    get_listed_df = pd.read_csv("./resources/get_listed.csv")

    df = pd.concat([unlisted_df, get_listed_df], ignore_index=True)
    header = ["company_name", "industry", "listing_date", "establishment_date"]
    docs = df.apply(lambda row: f"{row['company_name']} {row['industry']} {row['listing_date']} {row['establishment_date']}", axis=1).tolist()
    docs.insert(0, " ".join(header))

    documents = [Document(page_content=doc) for doc in docs]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=vector_store_path
    )

    return vector_store

def initialize_retriever() -> VectorStoreRetriever:
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()


def initialize_chain() -> RunnableSequence:
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI()
    retriever = initialize_retriever()
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    return chain


@app.post("/ask")
async def ask_question(query: QueryRequest):
    chain = initialize_chain()
    result = chain.invoke(query.question)
    return {"answer": result.content}
