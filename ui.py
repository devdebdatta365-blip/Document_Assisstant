import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import tempfile
import hashlib

load_dotenv()

# Page config
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 DocuMind AI")
st.caption("Chat with your documents intelligently")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# Sidebar
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Load PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(pages)

                # Generate unique collection name per PDF
                collection_name = hashlib.md5(
                    uploaded_file.name.encode()
                ).hexdigest()[:8]

                # Store collection name in session
                st.session_state.collection_name = collection_name

                # Create ChromaDB with unique collection
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=st.session_state.embeddings,
                    persist_directory="./chroma_db",
                    collection_name=collection_name
                )

                # Clear old chat history
                st.session_state.messages = []
                st.session_state.chat_history = []

                # Clean up temp file
                os.unlink(tmp_path)

                st.success(f"✅ PDF Processed!")
                st.info(f"📊 Total chunks: {len(chunks)}")
                st.info(f"📄 Total pages: {len(pages)}")
                st.rerun()

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + Groq + ChromaDB")

# Main chat area
if st.session_state.vector_store is None:
    st.info("👈 Please upload a PDF from the sidebar to get started!")

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for page in message["sources"]:
                        st.write(f"📄 Page {page}")

    # Chat input
    question = st.chat_input("Ask anything about your document...")

    if question:
        # Show human message immediately
        with st.chat_message("human"):
            st.write(question)

        # Add to messages
        st.session_state.messages.append({
            "role": "human",
            "content": question
        })

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Setup LLM
                llm = ChatGroq(model="llama-3.1-8b-instant")

                # Retrieve relevant chunks
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                relevant_docs = retriever.invoke(question)

                # Combine chunks into context
                context = "\n\n".join([
                    doc.page_content for doc in relevant_docs
                ])

                # Create prompt with memory
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful assistant.
Answer the question based on the context below.
If the answer is not in the context, say 'I don't know'.
Be as detailed as possible.

Context:
{context}"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                # Get answer
                chain = prompt | llm
                answer = chain.invoke({
                    "context": context,
                    "question": question,
                    "chat_history": st.session_state.chat_history
                })

                # Show answer
                st.write(answer.content)

                # Collect unique source pages
                seen_pages = []
                source_pages = []
                for doc in relevant_docs:
                    page = doc.metadata['page']
                    if page not in seen_pages:
                        seen_pages.append(page)
                        source_pages.append(page)

                # Show sources
                with st.expander("📚 Sources"):
                    for page in source_pages:
                        st.write(f"📄 Page {page}")

        # Update chat history
        st.session_state.chat_history.append(
            HumanMessage(content=question)
        )
        st.session_state.chat_history.append(
            AIMessage(content=answer.content)
        )

        # Save to messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.content,
            "sources": source_pages
        })