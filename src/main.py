import sys
import argparse
from pathlib import Path

import colorama
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from local_config import MODEL_PATH

def create_directory_loader(directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*pdf",
        loader_cls=PyMuPDFLoader,
    )

def main(args):

    colorama.init()

    path_fmt = Path(args.directory).resolve()

    # Get documents
    #https://github.com/langchain-ai/langchain/issues/9749
    pdf_loader = create_directory_loader(path_fmt)
    pages = pdf_loader.load_and_split()

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(pages)

    # Create embeddings
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    args = parser.parse_args()
    return args

if __name__=="__main__":

    args = parse_args()

    main(args)