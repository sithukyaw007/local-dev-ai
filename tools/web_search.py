"""Web search tool using Tavily API for real-time information retrieval."""

import os

from tavily import TavilyClient


# OpenAI-compatible tool schema for function calling
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time information. "
            "Use this for questions about current events, weather, news, "
            "live data, or anything that requires up-to-date information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                }
            },
            "required": ["query"],
        },
    },
}


def execute(query: str) -> str:
    """Execute a web search via Tavily and return formatted results."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=5)

    results = []
    for item in response.get("results", []):
        title = item.get("title", "No title")
        content = item.get("content", "No content")
        url = item.get("url", "")
        results.append(f"**{title}**\n{content}\nSource: {url}")

    if not results:
        return f"No results found for: {query}"

    return "\n\n---\n\n".join(results)
