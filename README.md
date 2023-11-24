# document-qna-chatbot
## Chatbot for asking questions about documents using open-source tools.  Uses RAG with Langchain.

### Background
Many tutorials on LangChain chatbots rely on using paid tokens from 3rd party providers.
Our aim is to construct a simple document question answering bot using open-source
tools.

The model architecture is based on a ConversationalRetrievalChain with memory.

We tested with the following data:
* csv - health record [data](https://www.kaggle.com/datasets/louiscia/transcription-samples-mtsamples) available from Kaggle \
    Where the source column of text is 'Transcription'
* pdf
    Several academic papers on using machine learning for automatic code generation.

### How to run
1. Codebase uses pipenv as environment manager.  Install pipenv\
`pip install pipenv`

2. Navigate to the project directory.

3. Run command \
`pipenv install --ignore-pipfile`

You can also of course install the required packages with your preferred
environment manager.  The package list is found in Pipfile.

4. Make sure you have a directory with the target documents that you want to
ask questions about.  At present, pdf and csv formats supported.

5. Download an open-source language model.
In this code, we used [GPT4All](https://gpt4all.io/index.html). \
If you use [other models](https://python.langchain.com/docs/guides/local_llms#llms), \
you must change the llm and embeddings imports to the appropriate one: \
`from langchain.llms import GPT4All` \
`from langchain.embeddings import GPT4AllEmbeddings`

Refer [here](https://api.python.langchain.com/en/latest/api_reference.html#module-langchain.llms) for available LLMs.

5. Create a `local_config.py` file with a MODEL_PATH variable pointing to your model.

6. Navigate to your src directory and run: \
`python main.py --directory '<document_path>' --file_type 'csv'`

### Improvements
Here are some ideas for improvements.  We list them in order of priority:

1. Performance improvement.
Currently, a relatively involved question takes around 10 minutes. \
Yes, totally impractical but this is a first attempt on open-source data running
on consumer-grade hardware.

2. Document reference improvement.
In the case of CSVs, the output includes the source rows of the answer, \
but there appears to be a bug with row duplication.  Need to test how this works \
for multiple csv documents.

3. More advanced pdf processing.
It is practical to be able to handle PDFs with charts and tables.




