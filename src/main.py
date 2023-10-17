import argparse
from pathlib import Path
import time

import colorama
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from local_config import MODEL_PATH

def create_directory_loader(file_type, directory_path):

    loaders = {
    'pdf': PyMuPDFLoader,
    #'.xml': UnstructuredXMLLoader,
    'csv': CSVLoader,
    }

    # # For some reason get encoidng error with default csv settings
    # # Explicitly specify utf8 encoding
    loader_kwargs = {'csv': {'encoding': 'utf-8', 'source_column': 'transcription'},
                     'pdf': None}

    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_kwargs = loader_kwargs[file_type],
        loader_cls = loaders[file_type],
    )

def main(args):

    colorama.init()

    path_fmt = Path(args.directory).resolve()

    # Get documents
    #https://github.com/langchain-ai/langchain/issues/9749
    loader = create_directory_loader(args.file_type, path_fmt)
    pages = loader.load_and_split()

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(pages)

    # Create embeddings
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    # Create memory object
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key="answer")

    # Import LLM
    llm = GPT4All(
            model=MODEL_PATH,
            max_tokens=2048,
            )

    # Initialize conversation chain
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory, return_source_documents = True)

    while True:
        query = input(colorama.Fore.GREEN + "Input question: ")

        start_time = time.time()
        response = qa({"question": query})

        print(colorama.Fore.BLUE + response['answer'])
        if args.file_type=='csv':
            rows = [i.metadata['row'] for i in response['source_documents']]
            print(colorama.Fore.BLUE + f'Data source row numbers: {rows}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(colorama.Fore.YELLOW + f"Execution time: {elapsed_time:.3f} seconds")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', help='Directory of files')
    parser.add_argument('--file_type', help="Specify 'csv' or 'pdf'")
    args = parser.parse_args()
    return args

if __name__=="__main__":

    args = parse_args()

    main(args)