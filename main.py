import streamlit as st
from typing import List, TypedDict, Annotated
from pydantic import BaseModel
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import ArxivRetriever
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import ArxivRetriever
from operator import add
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set OpenAI API key
# os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# llm = ChatGroq(
#     model="llama-3.1-70b-versatile",
#     temperature=0.0,


# )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,

)

# Define the Paper model
class Paper(BaseModel):
    title: str
    abstract: str
    url: str

# Define the state using TypedDict to act as a dictionary
class ResearchFlowState(TypedDict):
    query: str
    limit: int
    semantic_scholar_results: Annotated[List[Paper], add]
    pubmed_results: List[Paper]
    arxiv_results: Annotated[List[Paper], add]
    insights: List[str]
    consensus_meter: str
    short_summary: str
    conclusion: str
    final_report: str

# Semantic Scholar Search Function
def search_semantic_scholar(state: ResearchFlowState):
    try:
        query = state["query"]
        tool = SemanticScholarQueryRun()
        semscholar_results = tool.run(query)

        papers = []
        paper_entries = semscholar_results.split("Published year:")
        for entry in paper_entries:
            if "Title:" in entry and "Abstract:" in entry:
                title = entry.split("Title:")[1].split("\n")[0].strip()
                abstract = entry.split("Abstract:")[1].strip()
                url = entry.split("URL:")[1].strip() if "URL:" in entry else "No URL available"
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "url": url
                })
        return {"semantic_scholar_results": papers}
    except Exception as e:
        st.error(f"Error during Semantic Scholar search: {e}")
        return {"semantic_scholar_results": []}

# PubMed Search Function
def search_pubmed(state: ResearchFlowState):
    try:
        query = state["query"]
        tool = PubmedQueryRun()
        pubmed_results = tool.run(query)

        papers = []
        if isinstance(pubmed_results, str):
            paper_entries = pubmed_results.split("Published:")
            for entry in paper_entries:
                if "Title:" in entry and "Summary::" in entry:
                    try:
                        title = entry.split("Title:")[1].split("\n")[0].strip()
                        abstract = entry.split("Summary::")[1].strip().split("\n")[0]
                        url = entry.split("URL:")[1].strip() if "URL:" in entry else "No URL available"
                        papers.append({
                            "title": title,
                            "abstract": abstract,
                            "url": url
                        })
                    except IndexError as e:
                        st.error(f"Error parsing entry: {entry}. Error: {e}")

            return {"pubmed_results": papers}
        return {"pubmed_results": []}

    except Exception as e:
        st.error(f"Error during PubMed search: {e}")
        return {"pubmed_results": []}

# Arxiv Search Function
def search_arxiv(state: ResearchFlowState):
    try:
        query = state["query"]
        retriever = ArxivRetriever(
            load_max_docs=state["limit"],
            get_ful_documents=False
        )
        arxiv_results = retriever.invoke(query)

        papers = []
        for doc in arxiv_results:
            papers.append({
                "title": doc.metadata.get("title", "No title available"),
                "abstract": doc.page_content[:500],
                "url": doc.metadata.get("entry_id", "No URL available")
            })

        return {"arxiv_results": papers}
    except Exception as e:
        st.error(f"Error during ArXiv search: {e}")
        return {"arxiv_results": []}

# Summarization and Insights Generation Function
def summarize_and_generate_insights(state: ResearchFlowState):
    try:
        papers = state["pubmed_results"] + state["semantic_scholar_results"] + state["arxiv_results"]
        papers = papers[:state["limit"]]

        if not papers:
            return {
                "short_summary": "No papers available for summarization.",
                "consensus_meter": "N/A",
                "insights": [],
                "conclusion": "No conclusion available due to lack of papers."
            }

        instruction_prompt = f"""
You are tasked with analyzing the following research papers on the topic: '{state["query"]}'.
Follow these steps to provide an organized and concise output:

1. **Summary**: Write a concise 100 word summary of the main findings from the research papers.
2. **Consensus Meter**: Just provide a consensus meter using the format:
   - Yes: X% 
   - Possibly: Y%
   - No: Z%
3. **Key Insights**: Highlight the most significant insights using concise headings and 3-5 sentences explanations. For each insight, include the title of the paper from which it is derived in parentheses. Each insight should be presented in 3-5 sentences, clearly describing the aspect of the research. Organize them as follows:
   - [Title of Insight] :
     [Short explanation] (Source: '[Title of Paper]')
   - [Title of Insight] :
     [Short explanation] (Source: '[Title of Paper]')
4. **Conclusion**: Provide a comprehensive conclusion summarizing the findings and overall stance of the papers on the topic almost 100 words.

Here are the papers:
"""
        # Include titles, abstracts, and titles for each paper
        for paper in papers:
            instruction_prompt += f"\n**Title:** {paper['title']}\n**Abstract:** {paper['abstract']}\n"

        # Ensure the prompt does not exceed token limits
        max_prompt_length = 3500
        if len(instruction_prompt) > max_prompt_length:
            instruction_prompt = instruction_prompt[:max_prompt_length]

        # Call the LLM using `invoke()` method
        llm_response = llm.invoke(instruction_prompt)

        # Access the content of the AIMessage object
        response_content = llm_response.content if isinstance(llm_response, AIMessage) else llm_response

        # Parse the response to extract summary, consensus meter, key insights, and conclusion
        summary = consensus_meter = conclusion = key_insights = "N/A"
        if "Consensus Meter:" in response_content and "Conclusion:" in response_content:
            parts = response_content.split("Consensus Meter:")
            summary = parts[0].strip()
            consensus_and_conclusion = parts[1].split("Conclusion:")
            consensus_meter = consensus_and_conclusion[0].split("Key Insights:")[0].strip()
            if "Key Insights:" in consensus_and_conclusion[0]:
                key_insights = consensus_and_conclusion[0].split("Key Insights:")[1].strip()
            conclusion = consensus_and_conclusion[1].strip() if len(consensus_and_conclusion) > 1 else "N/A"
        else:
            summary = response_content

        # Parse the key insights into a list for report formatting and include titles
        insights_list = []
        if key_insights != "N/A":
            insights_lines = key_insights.split("\n")
            for i, insight in enumerate(insights_lines):
                if insight.strip():
                    # Attach the title of the corresponding paper to the insight
                    paper_title = papers[i]["title"] if i < len(papers) else "No Title available"
                    insights_list.append(f"- {insight.strip()} (Source: '{paper_title}')")

        # Return the parsed information
        return {
            "short_summary": summary,
            "consensus_meter": consensus_meter.strip(),
            "insights": insights_list,
            "conclusion": conclusion.strip()
        }

    except Exception as e:
        st.error(f"Error during summarization and consensus: {e}")
        return {
            "short_summary": "Error in summarization.",
            "consensus_meter": "N/A",
            "insights": [],
            "conclusion": "N/A"
        }

#generate final report 
def generate_report(state: ResearchFlowState):
    # Format the final report in markdown
    report = f"# Final Research Report\n"
    report += f"## Query: {state['query']}\n"

    report += f"\n\n{state['short_summary']}\n"

    # Only add the consensus meter if it's not "N/A"
    if state['consensus_meter'] != "N/A" and state['consensus_meter'].strip():
        report += f"\n### Consensus Meter:\n**{state['consensus_meter']}**\n"

    # Add the key insights section only once
    if state["insights"]:
        report += "\n### Key Insights:\n"
        for insight in state["insights"]:
            report += f"{insight}\n"

    # Ensure the conclusion is displayed at the end
    if state['conclusion'] != "N/A" and state['conclusion'].strip():
        report += f"\n### Conclusion:\n{state['conclusion']}\n"

    # Return the final report in markdown format
    return {"final_report": report}



# Build the graph
builder = StateGraph(ResearchFlowState)
builder.add_node("search_semantic_scholar", search_semantic_scholar)
builder.add_node("search_pubmed", search_pubmed)
builder.add_node("search_arxiv", search_arxiv)
builder.add_node("summarize_and_consensus", summarize_and_generate_insights)
builder.add_node("generate_report", generate_report)

builder.add_edge(START, "search_pubmed")
builder.add_edge(START, "search_semantic_scholar")
builder.add_edge(START, "search_arxiv")
builder.add_edge("search_semantic_scholar", "summarize_and_consensus")
builder.add_edge("search_arxiv", "summarize_and_consensus")
builder.add_edge("search_pubmed", "summarize_and_consensus")
builder.add_edge("summarize_and_consensus", "generate_report")
builder.add_edge("generate_report", END)

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# Main Streamlit app
def main():
    st.title("Research Assistant Agent")

    query = st.text_input("Enter your research query:")
    limit = st.number_input("Number of results per source", min_value=1, max_value=20, value=10)

    if st.button("Run Research"):
        initial_state = ResearchFlowState(
            query=query,
            limit=limit,
            semantic_scholar_results=[],
            pubmed_results=[],
            arxiv_results=[],
            insights=[],
            consensus_meter="",
            short_summary="",
            conclusion="",
            final_report=""
        )

        thread = {"configurable": {"thread_id": "1"}}

        for event in graph.stream(initial_state, thread, stream_mode="values"):
            pass
            # st.write(event)  # Debugging line to see each event output

        final_state = event  # The last event should contain the final state
        
        # Check if 'final_report' is present in the final_state
        if "final_report" in final_state and final_state["final_report"]:
            st.markdown(final_state["final_report"], unsafe_allow_html=True)
        else:
            st.error("Final report generation failed. Please check the input or try again.")

if __name__ == "__main__":
    main()