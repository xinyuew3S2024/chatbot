import requests
import json
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os


SERP_API_KEY = os.environ.get("SERP_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_weather_serp(query: str) -> str:
    """
    Call the Scale SERP API using the provided query and try to extract detailed
    temperature forecast information if available.
    """
    params = {
        'api_key': SERP_API_KEY, 
        'q': query,  # The query should include any location and day details.
        'gl': 'us',
        'hl': 'en',
        'google_domain': 'google.com',
        'include_ai_overview': 'true'
    }
    response = requests.get('https://api.scaleserp.com/search', params=params)
    if response.status_code == 200:
        result_json = response.json()
        # Attempt to extract detailed forecast information.
        if "organic_results" in result_json and result_json["organic_results"]:
            first_result = result_json["organic_results"][0]
            # If the rich snippet exists and contains extensions, use those as detailed info.
            if "rich_snippet" in first_result and "top" in first_result["rich_snippet"]:
                top_info = first_result["rich_snippet"]["top"]
                if "extensions" in top_info and isinstance(top_info["extensions"], list):
                    detailed_forecast = " ".join(top_info["extensions"])
                    # Check if we have temperature info (e.g., containing "°F")
                    if "°F" in detailed_forecast:
                        return detailed_forecast
            # Fallback: try to extract from the snippet field.
            if "snippet" in first_result:
                snippet = first_result["snippet"]
                if "°F" in snippet:
                    return snippet
        # If extraction fails, return the full JSON as a fallback.
        return json.dumps(result_json, indent=2)
    else:
        return f"Error: Unable to fetch data from SERP API, status code: {response.status_code}"

# Wrap the API function as a LangChain Tool.
serp_weather_tool = Tool(
    name="SERPWeatherAPI",
    func=get_weather_serp,
    description=(
        "Retrieves live weather information using the Scale SERP API based on a user-provided query. "
        "Use this tool when the user asks about weather forecasts, and include detailed temperature data if available."
    )
)

# Initialize the language model.
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Initialize conversation memory.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create an agent with a more general but guiding prompt.
agent = initialize_agent(
    tools=[serp_weather_tool],
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": (
            "You are a helpful weather assistant. When a user asks about the weather, "
            "use the SERPWeatherAPI tool to obtain live weather forecast data. "
            "If the query asks for forecasts over multiple days, make sure to include detailed temperature information "
            "for each day in your final answer. Return your final answer as:\n\n"
            "Final Answer: <your concise and detailed forecast>\n\n"
            "Do not add extra commentary."
        )
    }
)

# Build the Streamlit web UI.
def main():
    st.title("Live Weather Chatbot")
    st.write(
        "Ask about the weather anywhere. For example:\n"
        "- What does the weather look like in Pittsburgh today?\n"
        "- How will the weather look in Pittsburgh over the next four days?"
    )
    
    user_query = st.text_input("Enter your weather query:")
    if st.button("Get Weather") and user_query:
        with st.spinner("Fetching live weather information..."):
            response = agent.run(user_query)
        st.write(response)

if __name__ == "__main__":
    main()