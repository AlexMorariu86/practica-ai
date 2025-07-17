#Cod chatbot, nu merge bine pe pdf-uri in romana

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os
import shutil

pdf_path = "C:/Users/senon/Desktop/SOAC_Matrix.pdf"

text = ""
with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(text)

embeddings = OllamaEmbeddings(model="mistral:instruct")

# Create vector store
vectorstore = Chroma.from_texts(
    texts=text_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


llm = OllamaLLM(model="mistral:instruct")

retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)


while True:
    user_question = input("\nuser: ")    
    response = retrieval_chain.invoke({"query": user_question})
    answer = response["result"]
    print(f"\nchatbot: {answer}")
