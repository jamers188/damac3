import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# load environment variables
load_dotenv()


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            # Add page number reference to help with context
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        else:
            text += f"\n--- Page {page_num + 1} (No extractable text) ---\n"
    return text


# get text chunks method with improved section handling
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    # Try to split by sections first to preserve section integrity
    import re
    
    # Look for section headers like "SECTION 1:" or "Section 1:" etc.
    section_pattern = re.compile(r'(?i)(SECTION\s+\d+[\s\-:]+.*?)(?=SECTION\s+\d+[\s\-:]+|$)')
    sections = section_pattern.findall(text)
    
    if sections:
        print(f"Found {len(sections)} sections in the document")
        text_chunks = []
        # Process each section as a chunk, or split further if too large
        for section in sections:
            if len(section) <= chunk_size:
                text_chunks.append(section)
            else:
                # If a section is too large, split it while preserving the section header
                header_match = re.match(r'(?i)(SECTION\s+\d+[\s\-:]+.*?)(\n|\r|\r\n)', section)
                if header_match:
                    header = header_match.group(1)
                    content = section[len(header):]
                    
                    # Split the content into chunks
                    pos = 0
                    while pos < len(content):
                        end_pos = min(pos + chunk_size, len(content))
                        # First chunk includes header
                        if pos == 0:
                            text_chunks.append(header + content[pos:end_pos])
                        else:
                            # Other chunks get a reminder of which section they belong to
                            text_chunks.append(f"(Continued from {header}) " + content[pos:end_pos])
                        pos += chunk_size - chunk_overlap
                else:
                    # Fallback if header pattern isn't clear
                    position = 0
                    while position < len(section):
                        start_index = max(0, position - chunk_overlap)
                        end_index = position + chunk_size
                        chunk = section[start_index:end_index]
                        text_chunks.append(chunk)
                        position = end_index - chunk_overlap
    else:
        # Fallback to original method if no sections found
        print("No clear sections found, using standard chunking")
        text_chunks = []
        position = 0
        while position < len(text):
            start_index = max(0, position - chunk_overlap)
            end_index = position + chunk_size
            chunk = text[start_index:end_index]
            text_chunks.append(chunk)
            position = end_index - chunk_overlap
            
    # Print the first few characters of each chunk for debugging
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i}: {chunk[:100]}...")
            
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
    # Use a better model that should still work on Streamlit Cloud
    model_id = "google/flan-t5-base"  # Better quality than small but still reasonable size
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        temperature=0.7,  # Add some temperature for more detailed responses
        top_p=0.95,       # Using nucleus sampling for better quality
        do_sample=True    # Enable sampling for more diverse outputs
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"Using HuggingFacePipeline with {model_id}")

    print("Creating conversation chain...")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Create a custom prompt template to help the model generate better responses
    template = """
    You are a helpful assistant that answers questions based on the provided documents.
    
    Context information from documents:
    {context}
    
    Chat History:
    {chat_history}
    
    Question: {question}
    
    Please provide a comprehensive, detailed answer based on the information in the documents:
    """
    
    QA_PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # Retrieve more context
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False,  # Don't return source documents to avoid memory conflict
        verbose=True
    )
    print("Conversation chain created")
    return conversation_chain


# handle user input method
def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        try:
            # Add prefixes to encourage more detailed answers
            enhanced_question = f"Based on the documents provided, please give a detailed answer to this question: {user_question}"
            
            with st.spinner("Generating response... This may take a moment."):
                response = st.session_state.conversation({'question': enhanced_question})
                # Check if chat_history exists in the response
                if 'chat_history' in response:
                    st.session_state.chat_history = response['chat_history']
                else:
                    # If no chat_history in response, create or update it manually
                    if not st.session_state.chat_history:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response.get('answer', 'No response generated')})

                # Display chat history
                if isinstance(st.session_state.chat_history[0], dict):
                    # Handle dict format
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            st.write(user_template.replace(
                                "{{MSG}}", message["content"]), unsafe_allow_html=True)
                        else:
                            content = message["content"]
                            # Check if response is too short or unhelpful
                            if len(content.split()) < 5:
                                content += "\n\n(Note: The model provided a very brief response. Try rephrasing your question or uploading more detailed documents for better results.)"
                            st.write(bot_template.replace(
                                "{{MSG}}", content), unsafe_allow_html=True)
                else:
                    # Handle the original message format
                    for i, message in enumerate(st.session_state.chat_history):
                        if i % 2 == 0:
                            st.write(user_template.replace(
                                "{{MSG}}", message.content), unsafe_allow_html=True)
                        else:
                            content = message.content
                            # Check if response is too short or unhelpful
                            if len(content.split()) < 5:
                                content += "\n\n(Note: The model provided a very brief response. Try rephrasing your question or uploading more detailed documents for better results.)"
                            st.write(bot_template.replace(
                                "{{MSG}}", content), unsafe_allow_html=True)
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

    st.header("Fact Sheet :books:")
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
    
    # Create expandable section to show extracted text for debugging
    with st.expander("Document Processing Details", expanded=False):
        text_area = st.empty()

    for file in file_list:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".pdf":
            st.info(f"Processing {file.name}...")
            try:
                extracted_text = get_pdf_text(file)
                raw_text += extracted_text
                
                # Show first 1000 chars of extracted text for verification
                text_area.text_area(
                    f"Extracted Text Preview from {file.name} (first 1000 chars)",
                    extracted_text[:1000] + "...",
                    height=200
                )
            except Exception as e:
                st.warning(f"Could not process {file.name}: {str(e)}")
        elif file_extension == ".txt":
            st.info(f"Processing {file.name}...")
            extracted_text = file.getvalue().decode("utf-8")
            raw_text += extracted_text
            text_area.text_area(
                f"Extracted Text Preview from {file.name} (first 1000 chars)",
                extracted_text[:1000] + "...",
                height=200
            )
        elif file_extension == ".csv":
            st.info(f"Processing {file.name}...")
            extracted_text = file.getvalue().decode("utf-8")
            raw_text += extracted_text
            text_area.text_area(
                f"Extracted Text Preview from {file.name} (first 1000 chars)",
                extracted_text[:1000] + "...",
                height=200
            )
        else:
            st.error(f"File type {file_extension} not supported")
    
    if raw_text:
        st.success("Files processed successfully!")
        
        # Add a checkbox to enable section-specific processing
        use_section_search = st.checkbox("Enable section-specific search (recommended for forms and structured documents)", value=True)
        
        if use_section_search:
            # Look for sections in the text
            import re
            section_pattern = re.compile(r'(?i)(SECTION\s+\d+[\s\-:]+.*?)(?=SECTION\s+\d+[\s\-:]+|$)')
            sections = section_pattern.findall(raw_text)
            
            if sections:
                st.info(f"Found {len(sections)} sections in your document. You can search by section number.")
                # Display section headers for reference
                with st.expander("Document Sections Found", expanded=False):
                    for i, section in enumerate(sections):
                        # Extract just the header part
                        header = section.split('\n')[0] if '\n' in section else section[:100]
                        st.write(f"{i+1}. {header}")
        
        text_chunks = get_text_chunks(raw_text)
        st.info(f"Created {len(text_chunks)} text chunks for processing")
        
        with st.spinner("Creating knowledge base... This may take a few minutes."):
            vector_store = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_store)
        
        st.success("Knowledge base created! You can now ask questions about your documents.")
    else:
        st.error("No text was extracted from the uploaded files")


if __name__ == '__main__':
    main()
