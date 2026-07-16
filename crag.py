from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
load_dotenv()

#State
class GraphState(TypedDict):
    question:str
    documents: List[Document]
    generation: str
    web_search: bool
    steps: List[str] #tracks what path was taken - useful for UI Logging

# LLM

llm = ChatGroq(model="llama-3.1-8b-instant")

# Node 1- Retrieve
def retrieve(state,retriever):
    """
    Calls your existing hybrid+HYDE+rerank retriever.
    Writes documents to state
    """
    print("--- NODE: RETRIEVE ---")
    question =state["question"]
    documents,hypothetical_answer=retriever(question)
    return{
        "documents": documents,
        "question": question,
        "steps": state.get("steps",[])+["retrieve"]
    }


# Node 2: Grade documents
def grade_documents(state): 
    """
    Grades each retrieved document individually.
    Only marks web_search=True if ALL chunks are irrelevant.
    Partial relevance= still answer from PDF.
    """
    print("--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]

    grade_prompt = ChatPromptTemplate.from_template("""
You are grading whether a document chunk is relevant to a question.

Document:
{document}

Question: {question}

 Is this document chunk relevant to answering the question? Answer with a single word only — 'yes' if relevant, 'no' if not relevant.
""")

    grader_chain = grade_prompt | llm

    filtered_docs = []
    web_search_needed = False

    for doc in documents:
        result = grader_chain.invoke({
            "document": doc.page_content,
            "question": question
        })

        grade = result.content.strip().lower()
        if "yes" in grade:
            filtered_docs.append(doc)
        else:
            print(f"  ✗ Irrelevant chunk removed: {doc.page_content[:60]}...")
            #web_search_needed = True

    #Only trigger web search if less than 1 chunks passed grading
    
    if len(filtered_docs) <= 1:
        web_search_needed = True

    return {
        "documents": filtered_docs,
        "web_search": web_search_needed,
        "steps": state.get("steps", []) + ["grade_documents"]
    }
# Node 3 - Web Search

def web_search_node(state):
    """
    Falls back to Tavily web search when retrieved docs are irrelevant.
    Converts web results into Document objects and adds to state.
    """
    print("--- NODE: WEB SEARCH ---")
    question=state["question"]
    existing_docs=state["documents"]
    web_search_tool= TavilySearchResults(max_results=3)
    web_results=web_search_tool.invoke({
        "query":question
    })

    # Convert web results to Document objects
    web_docs=[
        Document(
            page_content=result["content"],
            metadata={
                "source": result["url"],
                "page":"web"
            }
        )
        for result in web_results
    ]

    #Combine any remaining good docs with web results
    all_docs=existing_docs+web_docs
    return{
        "documents": all_docs,
        "steps":state.get("steps",[])+["web_search"]
    }

# ── Node 4: Generate ─────────────────────────────

def generate(state):
    """
    Generates final answer using filtered documents.
    """
    print("---NODE: GENERATE----")
    question=state["question"]
    documents=state["documents"]
    context="\n\n".join([doc.page_content for doc in documents])
    prompt=ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based on the 
    context below. If the answer is not in the context, say 
    'I don't know'.
    Context:
    {context}
                                            
    Question: {question}
    Answer:
""")
    chain=prompt| llm
    result=chain.invoke({
        "context": context,
        "question": question
    })
    return{
        "generation": result.content,
        "steps": state.get("steps",[])+["generate"]
    }

# ── Conditional edge: decide next step after grading ─────
def decide_to_generate(state):
    """
    After grading, decide:
    - If web_search=True → go to web search node
    - If web_search=False → go directly to generate
    """
    if state["web_search"]:
        print("--- DECISION: Web search needed ---")
        return "web_search"
    else:
        print("--- DECISION: Documents sufficient, generating ---")
        return "generate"



# ── Build the graph ─────

def build_crag_graph(retriever):
    """
    Builds and compiles the CRAG LangGraph.
    Call this once after PDF is processed.

    """
    workflow=StateGraph(GraphState)
    #Add nodes
    workflow.add_node("retrieve",lambda s:retrieve(s,retriever))
    workflow.add_node("grade_documents",grade_documents)
    workflow.add_node("web_search",web_search_node)
    workflow.add_node("generate",generate)

    #Set entry point
    workflow.set_entry_point("retrieve")

    #Add edges
    workflow.add_edge("retrieve","grade_documents")

    #Conditional edge after grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search":"web_search",
            "generate":"generate"
        }
    )

    #Web searcg always leads to generate
    workflow.add_edge("web_search","generate")

    #Generate leads to END
    workflow.add_edge("generate",END)
    return workflow.compile()


