# contextSynth: Retrieval-Augmented Generator

contextSynth is a simple Retrieval-Augmented Generation (RAG) application that scrapes Wikipedia documents and answers relevant questions based on retrieved information. It leverages language models for embeddings, vector stores for indexing, and a retrieval-based question-answering system.

<div>
  <img src="./demo.jpg" alt="Preview">
</div>

## Features
- Web scraping of Wikipedia pages using `WebBaseLoader`
- Text splitting for efficient document chunking
- Embedding generation with `OllamaEmbeddings`
- Vector storage and retrieval using `Chroma`
- Cosine similarity calculations for relevance scoring
- Language model-based response generation with `ChatOllama`

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8) and install the required dependencies:

```sh
pip install -r requirements.txt
```

### Ollama Requirements
Ensure you have Ollama installed and set up for using `llama3`. You can install it using:

```sh
pip install ollama
```

## Usage

### 1. Indexing: Scrape and Embed Wikipedia Documents
```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Artificial_intelligence",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="mw-content-text")),
)

wiki_docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
)
splits = text_splitter.split_documents(wiki_docs)

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="llama3"),
)
```

### 2. Retrieval: Find Relevant Documents
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
question = "What is Artificial Intelligence?"
docs = retriever.get_relevant_documents(question)
```

### 3. Generation: Answer the Question
```python
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

template = """
    Answer the question based only on the following context:
    Context: {context}
    Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(model="llama3")
chain = prompt | llm
response = chain.invoke({"context": docs, "question": question})
print(response)
```

## Future Improvements
- Support for multiple sources beyond Wikipedia
- Enhanced ranking of retrieved documents
- Improved prompt engineering for better responses
