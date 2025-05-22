import os
import pandas as pd
from prophet import Prophet
from langgraph.graph import StateGraph, END
from src import analysis, insights_other
from src.llm import load_config
from langchain_ollama.llms import OllamaLLM
from typing import TypedDict, List, Dict, Any

# --- Setup (reuse your config and index logic) ---
cfg = load_config("config/config.json")
json_root = "data/json_input"
index, documents, embedder = analysis.build_product_index(json_root)

# Configure the LLM with essential parameters only
llm_client = OllamaLLM(
    model=cfg["model_name"],
    temperature=0.7,
    num_ctx=4096
)

# --- Node 1: Search ---
def search_node(state):
    query = state["query"]
    # If the query is short or just a brand, ask the LLM to expand it
    if len(query.split()) < 3:
        clarification_prompt = (
            f"The user asked: '{query}'. "
            "If this is a brand name or ambiguous, expand it to a full product query (e.g., 'sales of the beverage <brand>' or 'sales of <brand> across all categories'). "
            "Otherwise, return the query as is."
        )
        try:
            expanded_query = llm_client.invoke(clarification_prompt)
            if expanded_query and isinstance(expanded_query, str):
                query = expanded_query.strip()
        except Exception:
            pass  # fallback to original query if LLM fails

    # Now use the expanded query for vector retrieval
    best = analysis.search_vector_store(query, index, documents, embedder, top_k=1)[0]
    product = best["product"]
    
    # Extract time series data
    series = analysis.extract_product_timeseries(json_root, product)
    
    return {**state, "product": product, "series": series}

# --- Node 2: Forecast ---
def forecast_node(state):
    series = state.get("series", [])
    if len(series) < 2:
        return {**state, "observed": {"x": [], "y": []}, "forecast": {"x": [], "y": []}, "forecast_df": None}
    
    df = pd.DataFrame(series)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").rename(columns={"Date": "ds", "value": "y"})
    m = Prophet().fit(df)
    fut = m.make_future_dataframe(periods=14, freq="M")
    fc = m.predict(fut)
    fc_filt = fc[fc.ds > df.ds.max()]
    observed = {"x": df["ds"].dt.strftime("%Y-%m-%d").tolist(), "y": df["y"].tolist()}
    forecast = {"x": fc_filt["ds"].dt.strftime("%Y-%m-%d").tolist(), "y": fc_filt["yhat"].tolist()}
    return {**state, "observed": observed, "forecast": forecast, "forecast_df": fc_filt, "timeseries_df": df}

# --- Node 3: Insights ---
def insights_node(state):
    query = state["query"]
    product = state.get("product")
    forecast_df = state.get("forecast_df")
    timeseries_df = state.get("timeseries_df")
    series = state.get("series", [])
    
    insights = analysis.generate_insights_from_predictions(
        forecast_df[['ds','yhat']] if forecast_df is not None else None,
        product,
        series,
        lambda sys, usr: llm_client.invoke(f"{sys}\n\n{usr}")
    )
    return {**state, "insights": insights}

# --- Build the LangGraph pipeline ---
class AgentState(TypedDict, total=False):
    query: str
    product: str
    series: List[Dict]
    observed: Dict[str, List]
    forecast: Dict[str, List]
    forecast_df: Any
    timeseries_df: Any
    insights: str

graph = StateGraph(AgentState)
graph.add_node("search_data", search_node)
graph.add_node("generate_forecast", forecast_node)
graph.add_node("generate_insights", insights_node)
graph.add_edge("search_data", "generate_forecast")
graph.add_edge("generate_forecast", "generate_insights")
graph.add_edge("generate_insights", END)
graph.set_entry_point("search_data")
pipeline = graph.compile()

# --- Helper for Flask or CLI ---
def run_agentic_pipeline(query):
    state = {"query": query}
    result = pipeline.invoke(state)
    
    # Generate plots
    plots = analysis.generate_product_graphs(result.get("series", []), result.get("product", ""))
    
    # Add forecast to plots if available
    if result.get("forecast"):
        plots['forecast'] = {
            'title': f'Forecast for {result.get("product", "")}',
            'x_label': 'Date',
            'y_label': 'Predicted Value',
            'data': {
                'x': result["forecast"]["x"],
                'y': result["forecast"]["y"],
                'type': 'line',
                'name': 'Forecast'
            }
        }
    
    # Return the same structure as your normal mode
    return {
        'product': result.get("product", ""),
        'plots': plots,
        'insights': result.get("insights", "")
    }
