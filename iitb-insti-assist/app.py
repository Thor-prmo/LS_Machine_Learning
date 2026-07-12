import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#1. UI Setup 
st.set_page_config(page_title="IITB Insti-Assist", page_icon="🎓")
st.title("🎓 IITB Insti-Assist: Academic Bot")
st.write("Ask me anything about the IITB Undergraduate Rules & Regulations!")

#2. RAG Pipeline Initialization (Cached for performance) 
@st.cache_resource
def initialize_rag_pipeline():
    # 2a. Data Ingestion (Requires 5 PDFs in the 'data' folder)
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    if not documents:
        st.error("No documents found in the 'data' folder. Please add the UG Rule Book and 4 other PDFs.")
        st.stop()

    # 2b. Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 2c. Embedding & Vector Search (FAISS + all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2d. LLM API Integration (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.1 # Low temperature for factual responses
    )

    # 2e. Grounded Answering Prompt (Strict instruction to avoid hallucination)
    system_prompt = (
        "You are an academic assistant for IIT Bombay students. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not contained within the context below, you MUST say exactly: "
        "'I don't know.' Do not try to make up an answer or guess.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

#3. Interactive Web Interface
try:
    rag_chain = initialize_rag_pipeline()
except Exception as e:
    st.error(f"Error initializing the pipeline: {e}")
    st.stop()

# Initialize chat history (UI only, no conversational memory in LLM as per strict core requirements)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display source attribution if available
        if "sources" in message:
            with st.expander("View Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.caption(f"**Source {idx + 1}:** {source.metadata.get('source', 'Unknown Document')} (Page {source.metadata.get('page', 'N/A')})")
                    st.text(source.page_content[:200] + "...")

# Accept user input
if prompt := st.chat_input("E.g., What are the rules for academic probation?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching rule books..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            source_documents = response["context"]
            
            st.markdown(answer)
            
            # 2f. Source Attribution
            if source_documents and answer != "I don't know.":
                with st.expander("View Sources"):
                    for idx, doc in enumerate(source_documents):
                        st.caption(f"**Source {idx + 1}:** {doc.metadata.get('source', 'Unknown Document')} (Page {doc.metadata.get('page', 'N/A')})")
                        st.text(doc.page_content[:200] + "...")
                        
            # Save assistant response to UI history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": source_documents
            })
