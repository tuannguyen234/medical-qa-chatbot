import streamlit as st
import asyncio
from agents import Agent, Runner, function_tool
from pydantic import BaseModel

class PRArticle(BaseModel):
    article_text: str
    commentary: str

adult_writer_agent = Agent(
    name="Adult Writer Agent",
    instructions="""Write the article based on the information given that it is suitable for adults interested in culture. 
                    Be mature.""", 
    model="gpt-4o-mini",
)

teen_writer_agent = Agent(
    name="Teen Writer Agent",
    instructions="""Write the article based on the information given that it is suitable for teenagers who want to have a good time. 
                    Be cool!""", 
    model="gpt-4o-mini",
)

kid_writer_agent = Agent(
    name="Kid Writer Agent",
    instructions="""Write the article based on the information given that it is suitable for kids of around 8 years old. 
                    Be enthusiastic!""", 
    model="gpt-4o-mini",
)

format_agent = Agent(
    name="Format Agent",
    instructions=f"""Edit the article to add a title and subtitles and ensure the text is formatted as Markdown. Return only the text of article.""", 
    model="gpt-4o-mini",
)

researcher_agent = Agent(
    name="Research agent",
    instructions="""You are a Travel Agent who will find useful information for your customers of all ages.
                    Find information on the destination(s) given. 
                    When you have a result send it to the appropriate writer agent to produce a short PR text.
                    When you have the result send it to the Format agent for final processing.
                    """,
    model="gpt-4o-mini",
    tools = [kid_writer_agent.as_tool(
                tool_name="kids_article_writer",
                tool_description="Write an essay for kids",), 
            teen_writer_agent.as_tool(
                tool_name="teen_article_writer",
                tool_description="Write an essay for teens",), 
            adult_writer_agent.as_tool(
                tool_name="adult_article_writer",
                tool_description="Write an essay for adults",),
            format_agent.as_tool(
                tool_name="format_article",
                tool_description="Add titles and subtitles and format as Markdown",
        ),],
    output_type = PRArticle
)

async def run_agent(input_string):
    result = await Runner.run(researcher_agent, input_string)
    print("\n📦 Raw Responses:")
    for r in result.raw_responses:
        print(r)
    return result

# Streamlit UI

st.title("Travel Agent")
st.write("The travel agent will write about destinations for different audiences.")

destination = st.text_input("Enter a destination, select the age group and press 'Send':")
age_group = st.radio(
    "What age group is the reader?",
    ["Adult", "Teenager", "Child"],
    horizontal=True,
)

st.write("Response:")
response_container = st.container(height=500, border=True)

if st.button("Send"):
    response = asyncio.run(run_agent(f"The destination is {destination} and reader the age group is {age_group}"))
    with response_container:
        st.markdown(response.final_output.article_text)
    st.write(response)
    st.json(response.raw_responses)
import streamlit as st
import asyncio
from agents import Agent, Runner, function_tool
import wikipedia

@function_tool
def wikipedia_lookup(q: str) -> str:
    """Look up a query in Wikipedia and return the result"""
    return wikipedia.page(q).summary

research_agent = Agent(
    name="Research agent",
    instructions="""You research topics using Wikipedia and report on 
                    the results. """,
    model="gpt-4o-mini",
    tools=[wikipedia_lookup],
)

async def run_agent(input_string):
    result = await Runner.run(research_agent, input_string)
    print(f"Result: {result}")
    print("\n📦 Raw Responses:")
    for r in result.raw_responses:
        print(r)
    return result.final_output

# Streamlit UI

st.title("Simple Tool-using Agent")
st.write("This agent uses Wikipedia to look up information.")

user_input = st.text_input("Enter a query and press 'Send':")

st.write("Response:")
response_container = st.container(height=300, border=True)

if st.button("Send"):
    response = asyncio.run(run_agent(user_input))
    with response_container:
        st.markdown(response)

st.title("Simple Agent SDK Query")
print("Hello World")
# Create the agent instance
agent = Agent(name="Assistant",
              instructions="You are a helpful assistant",
              model="gpt-4o-mini")

# Input box for user query
user_input = st.text_input("Enter a query and press 'Send':")

# Container for the response
response_container = st.container(height=600, border=True)
st.write("Response:")

# Define async function to run the agent
async def get_agent_response(query):
    result = await Runner.run(agent, query)
    return result.final_output

# Handle button click
if st.button("Send"):
    if user_input.strip():
        # Streamlit does not support 'await' at the top level of a script.
        # You cannot use 'await' outside of an async function.
        # To run async code in Streamlit, use asyncio.run or nest_asyncio.
        response = asyncio.run(get_agent_response(user_input))
        with response_container:
            st.markdown(response)
    else:
        with response_container:
            st.markdown("*Please enter a query above and press 'Send'.*")

# streamlit run test.py --server.port 1000 --server.runOnSave true