from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import List

# Charger les variables d'environnement
load_dotenv()

# Initialiser FastAPI
app = FastAPI()

# Fonction pour extraire le texte du PDF
def get_pdf_text(pdf_docs: List[UploadFile]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Fonction pour découper le texte en morceaux
def get_text_chunks(text: str):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Fonction pour créer la store vectorielle à partir des chunks
def get_vectorstore(text_chunks: List[str]):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Fonction pour créer une chaîne de conversation avec récupération
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Endpoint pour télécharger des fichiers PDF et initialiser la chaîne de conversation
@app.post("/upload-pdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    # Extraire le texte brut des PDFs
    raw_text = get_pdf_text(files)
    
    # Diviser le texte en morceaux
    text_chunks = get_text_chunks(raw_text)
    
    # Créer la store vectorielle
    vectorstore = get_vectorstore(text_chunks)
    
    # Créer la chaîne de conversation
    conversation_chain = get_conversation_chain(vectorstore)
    
    # Sauvegarder la chaîne dans l'état global
    app.state.conversation_chain = conversation_chain
    
    return {"message": "PDFs processed and conversation chain created successfully."}

# Endpoint pour poser des questions à partir de la chaîne de conversation
@app.post("/ask-question/")
async def ask_question(question: str):
    if not hasattr(app.state, 'conversation_chain'):
        return {"error": "No conversation chain found. Please upload PDFs first."}

    # Gérer la question de l'utilisateur
    response = app.state.conversation_chain({'question': question})
    
    return {"response": response}
