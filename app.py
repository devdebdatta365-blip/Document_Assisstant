
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
# Step 1: Load PDF
print("Loading PDF...")
loader=PyPDFLoader("sample.pdf")
pages=loader.load()
print(f"Total pages: {len(pages)}")

#Step 2- Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.split_documents(pages)
print((f"Total chunks: {len(chunks)}"))

#Step 3 - Create Embeddings model
print("\nLoading embedding model")
embeddings=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

#Step 4 - Load or Create ChromaDB
if os.path.exists("./chroma_db"):
    #Load existing database
    print("Loading existing ChromaDB...")
    vector_store=Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    #Create new database
    print("Creating new ChromaDB...")
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
print("ChromaDB ready!")

#Step 5 - Setup LLM
llm= ChatGroq(model="llama-3.1-8b-instant")

#Step 6 - Create Retriever
retriever=vector_store.as_retriever(
    search_kwargs={"k":30}
)

#Step 7 - Create Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. 
Answer the question based on the context below.
If the answer is not in the context, say 'I don't know'.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),  # ← memory goes here
    ("human", "{question}")
])
chat_history = []
#Step 8 - Chat Loop
print("\n🤖 Document Assistant Ready!")
print("Type 'exit' to quit\n")
while True:
    question=input("You: ")

    if question.lower()=="exit":
        print("Goodbye!")
        break

    #Retrieve relevant chunks
    relevant_docs=retriever.invoke(question)

    


    #Combine chunks into context
    context=""
    for chunk in relevant_docs:
        context+=chunk.page_content+"\n"

    #Get answer from LLM
    response=prompt | llm
    answer=response.invoke({
        "context":context,
        "question":question,
        "chat_history":chat_history
    })
    # Update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer.content))
    
    
    #Show answer with sources
    print(f"\nAssistant: {answer.content}")
    print("\nSources:")
    seen_pages=[]
    for doc in relevant_docs:
        page=doc.metadata['page']
        if page not in seen_pages:
            print(f"- Page {page}")
            seen_pages.append(page) 
            
    print()