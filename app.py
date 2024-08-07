import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found. Please check your .env file.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not in the provided context. Please try asking another question."
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    return response["output_text"]

def main():
    st.set_page_config("Chat PDF", page_icon=":brain", layout="centered")
    st.header("Chat with PDF using Gemini💁")
    
    # Initialize chat session in Streamlit if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display the chat history
    for message in st.session_state.chat_history:
        st.markdown(message, unsafe_allow_html=True)
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        # Add user's message to chat history and display it
        st.session_state.chat_history.append(f"**Question:** {user_question}")
        st.markdown(f"**Question:** {user_question}", unsafe_allow_html=True)

        # Get the response from the model
        response_text = user_input(user_question)
        
        # Ensure proper spacing in the response text
        formatted_response_text = response_text.replace("\n", "<br>")
        
        # Add model's response to chat history and display it
        st.session_state.chat_history.append(f"**Answer:** {formatted_response_text}")
        st.markdown(f"**Answer:** {formatted_response_text}", unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

