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
import re
from operator import add
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import ArxivRetriever
import requests
from langchain.output_parsers import PydanticOutputParser
import time
from typing import List,Optional
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.0,


)

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.0,

# )


class Insight(BaseModel):
    title: str
    insight: str
    url: str


# Define the Paper model
class Paper(BaseModel):
    title: str
    abstract: str
    url: str

class ConsensusMeter(BaseModel):
    yes: Optional[str] = None
    possibly: Optional[str] = None
    no: Optional[str] = None

class Insights(BaseModel):
    title: str
    insight: str
    url: str

class LLMReponseScture(BaseModel):
    insights: List[Insights]
    consensus_meter: Optional[ConsensusMeter] = None
    short_summary: str
    conclusion: str


# Define the state using TypedDict to act as a dictionary
class ResearchFlowState(TypedDict):
    query: str
    r_query : str
    limit: int
    semantic_scholar_results: Annotated[List[Paper], add]
    pubmed_results: List[Paper]
    arxiv_results: Annotated[List[Paper], add]
    insights: List[Insights]
    consensus_meter: ConsensusMeter
    short_summary: str
    conclusion: str
    final_report: str

# Define the query refinement function
def refine_query(state: ResearchFlowState):
    try:
        # Original query from the state
        original_query = state["query"]

        # Instruction prompt for LLM to refine the query
        instruction_prompt = f"""
You are an expert at refining research queries. Your task is to correct any grammatical mistakes, complete incomplete phrases, and make the query more concise and to the point. Context of user query must be same .Ensure that the refined query maintains the original intent and clarity, optimized for fast and accurate results.

Original Query: "{original_query}"

Please respond with only the refined query text itself, without adding explanations or commentary.
        """

        # Call the LLM to refine the query
        llm_response = llm.invoke(instruction_prompt)
        refined_query=llm_response.content.strip()

        print(f"Refined Query: {refined_query}")

        return {"r_query": refined_query}
    except Exception as e:
        print(f"Error during query refinement: {e}")
        return {}




# Semantic Scholar Search Function
def search_semantic_scholar(state: ResearchFlowState):
    try:
        query = state["r_query"]
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": state["limit"],
            "fields": "title,abstract,url"
        }

        retries = 4  # Number of retry attempts
        delay = 5  # Initial delay in seconds

        for attempt in range(retries):
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                # Successful response
                data = response.json()
                papers = []
                for paper in data.get("data", []):
                    title = paper.get("title", "No title available")
                    abstract = paper.get("abstract", "No abstract available")
                    url = paper.get("url", "No URL available")

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "url": url
                    })
                return {"semantic_scholar_results": papers}

            elif response.status_code == 429:
                print(f"Rate limited by Semantic Scholar API. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff
            else:
                response.raise_for_status()  # Raise an error for other HTTP errors

        # If all retries failed
        print("Exceeded maximum retry attempts due to rate limiting.")
        return {"semantic_scholar_results": []}

    except Exception as e:
        print(f"Error during Semantic Scholar search: {e}")
        return {"semantic_scholar_results": []}


# PubMed Search Function
def search_pubmed(state: ResearchFlowState):
    try:
        query = state["r_query"]
        # print("pubmed queryyyyyyyyyy",query)
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Step 1: Search for PubMed article IDs using the query
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": state["limit"]
        }
        search_response = requests.get(base_url, params=search_params)
        search_data = search_response.json()

        # Extract PubMed IDs (PMIDs)
        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        # Step 2: Fetch details for each article using PubMed IDs
        if not pmids:
            print("No PubMed articles found.")
            return {"pubmed_results": []}

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        articles = fetch_response.text

        # Parse articles using XML for title, abstract, and construct URL
        from xml.etree import ElementTree as ET

        root = ET.fromstring(articles)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle", default="No title available")
            abstract = article.findtext(".//AbstractText", default="No abstract available")
            pmid = article.findtext(".//PMID", default="No ID available")

            # Construct the PubMed URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "No URL available"

            papers.append({
                "title": title,
                "abstract": abstract,
                "url": url
            })

        return {"pubmed_results": papers}

    except Exception as e:
        print(f"Error during PubMed search: {e}")
        return {"pubmed_results": []}



# Arxiv Search Function


def search_arxiv(state: ResearchFlowState):
    try:
        query = state["query"]
        # print("arxiv queryyyyyyyyyy",query)
        retriever = ArxivRetriever(
            load_max_docs=state["limit"],
            get_ful_documents=False
        )
        arxiv_results = retriever.invoke(query)

        papers = []
        for doc in arxiv_results:
            entry_id = doc.metadata.get("Entry ID", "")
            url = entry_id if entry_id else "No URL available"
            papers.append({
                "title": doc.metadata.get("Title", "No title available"),
                "abstract": doc.page_content[:500],
                "url": url
            })

        return {"arxiv_results": papers}
    except Exception as e:
        print(f"Error during ArXiv search: {e}")
        return {"arxiv_results": []}
    

def summarize_and_generate_insights(state: ResearchFlowState):
    try:
        papers = state["arxiv_results"] + state["semantic_scholar_results"] + state["pubmed_results"]
        papers = papers[:state["limit"]]

        if not papers:
            return {
                "short_summary": "No papers available for summarization.",
                "consensus_meter": "N/A",
                "insights": [],
                "conclusion": "No conclusion available due to lack of papers."
            }

        instruction_prompt = f"""
You are an expert assistant tasked with analyzing research papers on the topic: **'{state["r_query"]}'**.

### Instructions:

1. **Summary**: In `short_summary`, provide a brief overview of the main findings. Begin with a concise definition of the topic, followed by a synthesis of findings from the research papers.

2. **Consensus Meter**: If the query is specific, populate the `consensus_meter` object with the estimated level of agreement on the topic in percentages, formatted as follows:
   - **Yes**: X%
   - **Possibly**: Y%
   - **No**: Z%

3. **Key Insights**: Identify 6 to 8 notable insights from the research, adding each in the `insights` list with:
    - `title`: A brief, descriptive title for the insight
    - `insight`: A brief explanation of the insight
    - `url`: Source link for the research paper

4. **Conclusion**: In `conclusion`, state["query"] summarize the overall stance and any key implications from the research.

### Research papers:



Please follow these steps to provide a well-organized, concise, and professional report in the specified JSON format below:

### Output Format:

```json
{{
    "insights": [
      {{
        "title": "string",
        "insight": "string",
        "url": "string"
      }}
    ],
    "consensus_meter": {{null if general query else {{
      "yes": "string",
      "possibly": "string",
      "no": "string"
    }}}},
    "short_summary": "string",
    "conclusion": "string"
  }}
```a

### Research papers:

"""



        # Include titles, abstracts, and URLs for each paper
        for paper in papers:
            instruction_prompt += f"\n**Title:** {paper['title']}\n**Abstract:** {paper['abstract']}\n"
            if paper.get("url") and paper["url"] != "No URL available":
                instruction_prompt += f"**URL:** {paper['url']}\n"

        # max_prompt_length = 3500
        # if len(instruction_prompt) > max_prompt_length:
        #     instruction_prompt = instruction_prompt[:max_prompt_length]

        llm_response = llm.invoke(instruction_prompt).content

        response_content = PydanticOutputParser(pydantic_object=LLMReponseScture).parse(llm_response)
        print(response_content)

        return {
            "short_summary": response_content.short_summary,
            "consensus_meter": response_content.consensus_meter,
            "insights": response_content.insights,
            "conclusion": response_content.conclusion
        }

    except Exception as e:
        print(f"Error during summarization and consensus: {e}")
        return {
            "short_summary": "Error in summarization.",
            "consensus_meter": "N/A",
            "insights": [],
            "conclusion": "N/A"
        }
def generate_report(state: ResearchFlowState):
    # Start with a report title and query
    report = "# 📘 Final Research Report\n\n"
    report += f"**🔍 Query**: *{state['query'].strip()}*\n\n"

    # Short summary section with a divider
    report += "---\n## 📝 Short Summary:\n\n"
    report += f"{state['short_summary']}\n\n"

    # Display consensus meter, if available
    if state['consensus_meter']:
        report += "## 📊 Consensus Meter:\n\n"
        report += f"- **Yes**: {state['consensus_meter'].yes} \n"
        report += f"- **Possibly**: {state['consensus_meter'].possibly} \n"
        report += f"- **No**: {state['consensus_meter'].no} \n\n"
    # else:
    #     report += "## 📊 Consensus Meter:\n\n"
    #     report += "The consensus meter is not available for this query.\n\n"



    urls = []  # List to collect URLs

    # Key Insights section in a clear, organized bullet format
    if state["insights"]:
        report += "## 💡 Key Insights:\n\n"

        # Loop through each insight and format according to the required structure
        for insight in state["insights"]:
            # Title of the insight as a bullet point under Key Insights
            report += f"- **{insight.title}:**\n"
            # Insight description as a nested bullet point
            report += f"  - {insight.insight}\n"
            # Actual source link as a nested bullet point
            report += f"  - {insight.url}\n\n"

            urls.append(insight.url)  # Append the URL to the list

        report += "\n"

    # Conclusion section with clear formatting
    if state['conclusion'] != "N/A" and state['conclusion'].strip():
        report += "## 🔍 Conclusion:\n\n"
        report += f"{state['conclusion']}\n\n"

    # Display all collected URLs at the end of the report
    if urls:
        report += "\n## 🔗 Sources:\n\n"
        for url in urls:
            report += f"- {url}\n"

    # Closing line
    report += "---\n*End of Report*\n"

    return {"final_report": report}


# Build the graph
builder = StateGraph(ResearchFlowState)

# # Add nodes
builder.add_node("refine_query", refine_query)
builder.add_node("search_semantic_scholar", search_semantic_scholar)
builder.add_node("search_pubmed", search_pubmed)
builder.add_node("search_arxiv", search_arxiv)
builder.add_node("summarize_and_consensus", summarize_and_generate_insights)
builder.add_node("generate_report", generate_report)

# Define edges for parallel execution
builder.add_edge(START, "refine_query")
builder.add_edge("refine_query", "search_pubmed")
builder.add_edge("refine_query", "search_semantic_scholar")
builder.add_edge("refine_query", "search_arxiv")

# Make sure both search nodes lead to summarization
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
    st.title("Research Paper Analysis Tool")
    st.markdown("This tool allows you to input a research query and retrieve summarized insights from various research sources (Semantic Scholar, PubMed, ArXiv).")

    query = st.text_input("Enter your research query:")
    limit = st.number_input("Number of results per source", min_value=1, max_value=20, value=10)

    if st.button("Generate Research Report"):
        initial_state = ResearchFlowState(
            query=query,
            limit=limit,
            r_query="",
            semantic_scholar_results=[],
            pubmed_results=[],
            arxiv_results=[],
            insights=[],
            consensus_meter="",
            short_summary="",
            conclusion="",
            final_report=""
        )

        # Set up a unique thread for this run
        thread = {"configurable": {"thread_id": "1"}}

        # Track progress
        # progress_text = st.empty()
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