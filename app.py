import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# load environment variables
load_dotenv()


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() for page in pdf_reader.pages)


# get text chunks method
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_chunks = []
    position = 0
    # Iterate over the text until the entire text has been processed
    while position < len(text):
        start_index = max(0, position - chunk_overlap)
        end_index = position + chunk_size
        chunk = text[start_index:end_index]
        text_chunks.append(chunk)
        position = end_index - chunk_overlap
    return text_chunks


# get vector store method
def get_vectorstore(text_chunks):
    try:
        # Use a smaller, more reliable model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Using HuggingFaceEmbeddings with all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error with HuggingFaceEmbeddings: {e}")
        # Try another smaller model
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
            print("Using HuggingFaceEmbeddings with paraphrase-MiniLM-L3-v2")
        except Exception as e:
            print(f"Error with fallback embeddings: {e}")
            raise Exception("Failed to initialize embeddings. Please check your dependencies.")
    
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print(f"Vector store created with {type(embeddings).__name__}")
    return vector_store


# get conversation chain method
def get_conversation_chain(vectorstore):
    # Use a lightweight model that should work on Streamlit Cloud
    model_id = "google/flan-t5-small"  # A smaller model that should run on most systems
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"Using HuggingFacePipeline with {model_id}")

    print("Creating conversation chain...")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    print("Conversation chain created")
    return conversation_chain


# handle user input method
def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            st.info("Try asking a simpler question or processing different documents.")
    else:
        st.write("Please upload PDFs and click process")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.subheader("Your Documents")

    # init sidebar
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click process", type="pdf", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs... This may take a few minutes."):
                    try:
                        process_files(pdf_docs, st)
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
                        st.info("Try uploading smaller PDF files or fewer documents.")
            else:
                st.error("Please upload at least one PDF file")


def process_files(file_list, st):
    raw_text = ""

    for file in file_list:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".pdf":
            st.info(f"Processing {file.name}...")
            try:
                raw_text += get_pdf_text(file)
            except Exception as e:
                st.warning(f"Could not process {file.name}: {str(e)}")
        elif file_extension == ".txt":
            st.info(f"Processing {file.name}...")
            raw_text += file.getvalue().decode("utf-8")
        elif file_extension == ".csv":
            st.info(f"Processing {file.name}...")
            raw_text += file.getvalue().decode("utf-8")
        else:
            st.error(f"File type {file_extension} not supported")
    
    if raw_text:
        st.success("Files processed successfully!")
        text_chunks = get_text_chunks(raw_text)
        st.info(f"Created {len(text_chunks)} text chunks")
        
        with st.spinner("Creating knowledge base... This may take a few minutes."):
            vector_store = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_store)
        
        st.success("Knowledge base created! You can now ask questions about your documents.")
    else:
        st.error("No text was extracted from the uploaded files")


if __name__ == '__main__':
    main()
