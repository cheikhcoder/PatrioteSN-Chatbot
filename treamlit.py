import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
        length_function= len    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)





def main():
    load_dotenv()
    st.set_page_config(page_title="CitoyenSN",page_icon=":books:")
    st.header("CitoyenSN")
    user_question = st.text_input("pose une question sur le droit senegalais ") 
    if user_question:
        handle_user_question(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    with st.sidebar:
        st.subheader("tes documents")
        pdf_docs = st.file_uploader("upload your pdf",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing"):
                #get pdf raw text
                raw_text = get_pdf_text(pdf_docs)
                #get the text chunks 
                text_chunks = get_text_chunks(raw_text)
                # create the vector store 
                vecorstore = get_vectorstore(text_chunks)
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vecorstore)


              




if __name__ == "__main__":
    main()
