import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.google import Gemini
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools
import os

st.title("Multi-Agent AI Researcher 🔍🤖")
st.caption("Recherche sur HackerNews et le web, puis rédaction de rapports et de posts.")

MODEL_ID = "gemini-2.5-flash"  # à vérifier selon les modèles disponibles sur ton compte

# Clé Google : Secrets Streamlit en priorité, sinon saisie manuelle
google_api_key = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""
if not google_api_key:
    google_api_key = st.text_input("Google API Key", type="password")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    hn_researcher = Agent(
        name="HackerNews Researcher",
        model=Gemini(id=MODEL_ID),
        role="Gets top stories from hackernews.",
        tools=[HackerNewsTools()],
    )

    web_searcher = Agent(
        name="Web Searcher",
        model=Gemini(id=MODEL_ID),
        role="Searches the web for information on a topic",
        tools=[DuckDuckGoTools()],
        add_datetime_to_context=True,
    )

    article_reader = Agent(
        name="Article Reader",
        model=Gemini(id=MODEL_ID),
        role="Reads articles from URLs.",
        tools=[Newspaper4kTools()],
    )

    hackernews_team = Team(
        name="HackerNews Team",
        model=Gemini(id=MODEL_ID),
        members=[hn_researcher, web_searcher, article_reader],
        instructions=[
            "First, search hackernews for what the user is asking about.",
            "Then, ask the article reader to read the links for the stories to get more information.",
            "Important: you must provide the article reader with the links to read.",
            "Then, ask the web searcher to search for each story to get more information.",
            "Finally, provide a thoughtful and engaging summary.",
        ],
        markdown=True,
        debug_mode=False,
        show_members_responses=True,
    )

    query = st.text_input("Entre ta requête de rapport")

    if query:
        with st.spinner("Recherche en cours..."):
            response: RunOutput = hackernews_team.run(query, stream=False)
        st.write(response.content)
