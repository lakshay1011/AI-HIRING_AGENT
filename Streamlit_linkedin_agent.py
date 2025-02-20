import streamlit as st
from phi.model.groq import Groq
from phi.agent import Agent
from phi.tools.googlesearch import GoogleSearch
from phi.tools.crawl4ai_tools import Crawl4aiTools
import io
import sys
import re
import pandas as pd

# Function to remove ANSI escape codes and unnecessary characters
def clean_response(text):
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_text = ansi_escape.sub('', text)
    
    # Remove unnecessary characters like '┃'
    cleaned_text = cleaned_text.replace('┃', '').strip()
    return cleaned_text

def parse_response_to_table(response):
    # Split the response into lines
    lines = response.strip().split('\n')
    
    # Ensure we remove any accidental column name repetitions
    expected_columns = ["Name", "Profile URL", "Experience", "Location", "Skillset", "Confidence Score"]
    rows = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            # Split the line into columns using regex to handle varying spaces
            columns = re.split(r'\s{2,}', line.strip())  # Split by 2 or more spaces
            
            # Ensure LinkedIn URLs remain intact
            if len(columns) >= 5 and not any(col in expected_columns for col in columns):  
                # Sometimes, profile URLs break due to spaces; reconstruct them if needed
                name, profile_url, experience, location, skillset = columns[:5]
                confidence_score = columns[5] if len(columns) > 5 else "N/A"
                
                # Ensure the profile URL starts with "http" or "www"
                if not profile_url.startswith("http"):
                    profile_url = columns[1] + " " + columns[2]  # Reconstructing truncated URLs
                    
                rows.append({
                    "Name": name,
                    "Profile URL": profile_url.strip(),
                    "Experience": experience,
                    "Location": location,
                    "Skillset": skillset,
                    "Confidence Score": confidence_score
                })

    # Convert the list of rows into a Pandas DataFrame
    return pd.DataFrame(rows, columns=expected_columns)


# LINKEDIN SEARCH AGENT (Using Google Search for LinkedIn Profiles)
linkedin_search_agent = Agent(
    name="LinkedIn Search Agent",
    description="You are a LinkedIn search agent that helps users find relevant profiles based on client requirements.",
    instructions=[
        "Always include profile URLs and key details.",
        "Provide a confidence score (1-100) based on relevance to the query.",
        "Include a column for skillset extracted from the profile."
    ],
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[GoogleSearch()],
    show_tool_calls=True,
    debug_mode=True,
)

# WEB CRAWLER AGENT
web_crawler_agent = Agent(
    name="Web Crawler Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[Crawl4aiTools(max_length=None)],
    show_tool_calls=True,
    markdown=True,
    description="You are a web crawler that fetches detailed information from LinkedIn profile URLs.",
    instructions=[
        "Ensure all extracted details are accurate and well-structured.",
        "Extract the candidate's skillset and include it in the table.",
        "Assign a confidence score (1-100) based on completeness and accuracy of extracted information."
    ],
)

# AI HIRING AGENT TEAM
hiring_agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[linkedin_search_agent, web_crawler_agent],
    instructions=[
        "Always include profile URLs and extracted details and include complete profile URL.",
        "Use tables to display candidate information clearly.",
        "Each row should contain: Name, Profile URL, Experience, Location, Skillset, and Confidence Score (1-100). Exclude adding column names in the first row."
    ],
    show_tool_calls=True,
    markdown=True
)

# Function to capture print output
def capture_print_response(query):
    # Capture the print statements into a string buffer
    buffer = io.StringIO()
    sys.stdout = buffer
    hiring_agent_team.print_response(query)
    sys.stdout = sys.__stdout__  # Reset stdout
    return buffer.getvalue()

# Streamlit UI for input and display results
st.title('LinkedIn Profile Search and Crawler')
st.write("Enter a query to search for LinkedIn profiles of candidates based on your requirements.")

# User input for query
query = st.text_input("Enter the search query (e.g., HR Manager of XYZ Company, Jaipur, Rajasthan):")

if query:
    # Display the search results based on the input query
    with st.spinner('Searching for profiles...'):
        response = capture_print_response(query)
        
        # Clean the response by removing ANSI escape codes and unnecessary characters
        cleaned_response = clean_response(response)
        print(cleaned_response)
        
        # Parse the cleaned response into a structured table
        df = parse_response_to_table(cleaned_response)
        
        # Check if the DataFrame is empty
        if not df.empty:
            st.write("Results:")
            st.dataframe(df)  # Display the results in a table format
        else:
            st.write("No results found. Please try a different query.")
    
    # Additional information about the agents can be displayed
    st.subheader("Agent Information")
    st.write(f"LinkedIn Search Agent: {linkedin_search_agent.description}")
    st.write(f"Web Crawler Agent: {web_crawler_agent.description}")
