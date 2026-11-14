import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()

if __name__ == "__main__":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]

    if pc.has_index(index_name):
        print(f"Index '{index_name}' found, deleting...")
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1536,  # OpenAIEmbeddings default (text-embedding-3-small)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created")

    loader = TextLoader(r"C:\Users\WALTON\Desktop\langchain-course\mediumblog1.txt")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20,
        separator="\n\n",
        length_function = len
    )

    chunks = text_splitter.split_documents(documents=docs)
    print(len(chunks))
