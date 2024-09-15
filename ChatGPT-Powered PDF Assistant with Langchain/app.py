import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

#Sidebar contents
with st.sidebar:
    st.title("LLM Chat App 🚀")
    st.markdown('''
                ## About
                This app is an LLM-powered chatbot built using:
                - [Streamlit](https://streamlit.io/)
                - [Langchain](https://python.langchain.com/)
                - [OpenAI](https://openai.com/) LLM model

                ''')
    st.write('\n' * 5)
    st.write('Made with ❤️ by Janeesha')


def main():
    st.header("Chat with the PDF 🤖")

    load_dotenv()

    #upload pdf files
    pdf=st.file_uploader("Upload PDF file", type=['pdf'])
    

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        
        #embeddings
        
        store_name=pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store=pickle.load(f)
            st.write("Vector store loaded from disk")
        else:
            embeddings=OpenAIEmbeddings()
            vector_store=FAISS.from_texts(chunks, embedding=embeddings)
            with open (f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        #accept user questions/ queries
        query=st.text_input("Ask me anything")

        if query:
            docs=vector_store.similarity_search(query=query, k=3)

            llm=OpenAI(temperature=0,)
            chain= load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs, question=query)
                print(cb)    
                
            st.write(response)

if __name__=='__main__':
    main()

