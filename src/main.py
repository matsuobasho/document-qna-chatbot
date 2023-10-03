import sys
import argparse

import colorama
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from local_config import MODEL_PATH


def main():

    colorama.init()

    # Get document
    loader = PyPDFLoader("./docs/NL2 code.pdf")
    pages = loader.load_and_split()

    # Create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    # Create memory object
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Import LLM
    llm = GPT4All(
            model=MODEL_PATH,
            max_tokens=2048,
            )

    # Initialize conversation chain
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

    while True:
        query = input(colorama.Fore.GREEN + "Input question: ")

        response = qa({"question": query})

        print(colorama.Fore.BLUE + response['answer'])

if __name__=="__main__":
    main()