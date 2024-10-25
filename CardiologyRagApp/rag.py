import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


# Load environment variables (API key)
load_dotenv()
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
#os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'YOUR_HUGGINGFACE_API_KEY'


# Set the path for your documents and Chroma database
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db"



# Load PDF documents from a directory
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

documents = load_documents()
print(f"Loaded {len(documents)} documents.\n")
print(documents[0])  # Preview one document for sanity check



# Split documents into manageable chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} page documents into {len(chunks)} chunks.")
    return chunks

chunks = split_text(documents)



# Preview the content of a chunk for sanity check
document = chunks[20]
print("\n")
print(document.page_content)
print("\n")
print(document.metadata)



# Initialize embeddings using PubMedBERT from Hugging Face Hub
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


# Save the chunks to a Chroma vector database
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Clean up existing DB to avoid conflicts
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )

    print(f"\nSaved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db

db = save_to_chroma(chunks)


repo_id="epfl-llm/meditron-7b"

llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=huggingface_api_key)


print("LLM Initialized....")

# Define the prompt template for the model
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])



# Create a retriever from the Chroma vectorstore
retriever = db.as_retriever(search_kwargs={"k": 4})  # Top 4 most relevant chunks



# Combine LLM with retriever in a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    return_source_documents=True, 
    chain_type_kwargs={"prompt": prompt}
)



# Example query for testing
query = "What are the common symptoms of cardiovascular diseases?"
response = qa_chain.invoke(query)



# Extract and display the answer and source documents
answer = response['result']
source_docs = response['source_documents']

print(f"Answer: {answer}")
for i, doc in enumerate(source_docs):
    print(f"Source Document {i+1}: {doc.page_content}")
    print(f"Source File: {doc.metadata['source']}")
