from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = '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'

# Create an OpenAI LLM (Replace with your API key)
llm = OpenAI(model_name="gpt-4-turbo", api_key=os.environ["OPENAI_API_KEY"])

# Set the SerpAPI API key
os.environ["SERPAPI_API_KEY"] = 'your_serpapi_api_key_here'


# Load tools (e.g., Wikipedia, Google Search)
tools = load_tools(["wikipedia", "serpapi"])  # Requires API keys for some tools

# Create the agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Run the agent
response = agent.run("Who won the FIFA World Cup in 2018?")
print(response)
