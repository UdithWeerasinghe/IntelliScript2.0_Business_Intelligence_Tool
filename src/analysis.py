##############normal code ####################

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def load_json_data(directory):
    json_files = find_json_files(directory)
    data = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data

def build_vector_store(categories):
    vectorizer = TfidfVectorizer(stop_words='english')
    category_embeddings = vectorizer.fit_transform(categories)
    return vectorizer, category_embeddings

# --- New FAISS Vector Store Functions ---

def build_faiss_index(output_directory):
    """
    Loads JSON files from the output_directory and builds a FAISS index.
    Returns: (index, documents, embedder)
    Each document is a dict: {"text": ..., "metadata": {...}, "category": ...}
    """
    documents = []
    json_files = find_json_files(output_directory)
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        meta = content.get("metadata", {})
        data = content.get("data", {})
        for category in data.keys():
            # Create a document string combining the category and source metadata.
            doc_text = f"Category: {category}. File: {meta.get('file_name','')}. Sheet: {meta.get('sheet_name','')}."
            documents.append({"text": doc_text, "metadata": meta, "category": category})
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)
    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)
    print(f"FAISS index built with {index.ntotal} documents.")
    return index, documents, embedder

def search_vector_store(query, index, documents, embedder, top_k=1):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append(documents[idx])
    return results

def analyze_query_with_llm_faiss(user_query, generate_response_func, index, documents, embedder):
    system_prompt = "You are an expert data analyst. Help refine this query to better match data categories."
    refined_query = generate_response_func(system_prompt, user_query)
    results = search_vector_store(refined_query, index, documents, embedder, top_k=1)
    if results:
        best_match = results[0]
        print(f"Most relevant category: {best_match['category']}")
        print(f"Found in file: {best_match['metadata'].get('file_name')} sheet: {best_match['metadata'].get('sheet_name')}")
        return best_match['category']
    else:
        return None

def extract_values_for_category(json_data, category):
    values = []
    for file_data in json_data:
        data = file_data.get("data", {})
        if category in data:
            values.extend(data[category])
    return values

def plot_trend_and_save(values, category, output_file):
    dates = []
    y_values = []
    for entry in values:
        dates.append(entry["Date"])
        y_values.append(entry["value"])
    dates = pd.to_datetime(dates)
    df = pd.DataFrame({'ds': dates, 'y': y_values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=14, freq='M')
    forecast = model.predict(future)
    forecast_filtered = forecast[forecast['ds'] > max(dates)]
    plt.figure(figsize=(10,6))
    sns.lineplot(x=dates, y=y_values, label=f'{category} (Observed)', color='blue')
    sns.lineplot(x=forecast_filtered['ds'], y=forecast_filtered['yhat'],
                 label=f'{category} (Predicted)', color='orange', linestyle='--')
    if not forecast_filtered.empty:
        plt.plot([max(dates), forecast_filtered['ds'].iloc[0]],
                 [y_values[-1], forecast_filtered['yhat'].iloc[0]],
                 color='orange', linestyle='--')
    plt.title(f'Trend and Prediction for {category}')
    plt.xlabel('Time')
    plt.ylabel(category)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def display_predicted_values(values):
    dates = []
    y_values = []
    for entry in values:
        dates.append(entry["Date"])
        y_values.append(entry["value"])
    dates = pd.to_datetime(dates)
    df = pd.DataFrame({'ds': dates, 'y': y_values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=14, freq='M')
    forecast = model.predict(future)
    last_date = max(dates)
    forecast_filtered = forecast[forecast['ds'] > last_date]
    predicted_values = forecast_filtered[['ds', 'yhat']]
    print("Predicted Values for upcoming periods:")
    print(predicted_values.to_string(index=False))

# def generate_insights_from_predictions(predicted_values, parameter, generate_response_func):
#     summary_text = (
#         f"The parameter '{parameter}' has been analyzed with the following predicted values "
#         "for each month from the last observed date onward:\n\n"
#     )
    
#     summary_text += predicted_values.to_string(index=False, header=False)
#     summary_text += (
#         "\n\nProvide insights, recommendations, and suggestions to improve decision-making for this parameter."
#     )
#     system_prompt = (
#         "You are a highly skilled business consultant specializing in data-driven decision-making. "
#         "Analyze the provided predictions and generate actionable insights and recommendations."
#     )
#     insights = generate_response_func(system_prompt, summary_text, temperature=0.7, max_tokens=1000)
#     print(f"Insights and Recommendations for '{parameter}':\n")
#     print(insights)
#     return insights

###########3before agent working one

# def generate_insights_from_predictions(predicted_values, parameter, generate_response_func):
#     # If predicted_values is a list (raw Excel data), compute forecast first:
#     if isinstance(predicted_values, list):
#         dates = [entry["Date"] for entry in predicted_values]
#         y_values = [entry["value"] for entry in predicted_values]
#         dates = pd.to_datetime(dates)
#         df = pd.DataFrame({'ds': dates, 'y': y_values})
#         model = Prophet()
#         model.fit(df)
#         future = model.make_future_dataframe(periods=14, freq='M')
#         forecast = model.predict(future)
#         last_date = max(dates)
#         forecast_filtered = forecast[forecast['ds'] > last_date]
#         predicted_values_df = forecast_filtered[['ds', 'yhat']]
#     else:
#         predicted_values_df = predicted_values  # Assume it is a DataFrame

#     summary_text = (
#         f"The parameter '{parameter}' has been analyzed with the following predicted values "
#         "for each month from the last observed date onward:\n\n"
#     )
#     summary_text += predicted_values_df.to_string(index=False, header=False)
#     summary_text += (
#         "\n\nProvide insights, recommendations, and suggestions to improve decision-making for this parameter."
#     )
#     system_prompt = (
#         "You are a highly skilled business consultant specializing in data-driven decision-making. "
#         "Analyze the provided predictions and generate actionable insights and recommendations."
#     )
#     insights = generate_response_func(system_prompt, summary_text, temperature=0.7, max_tokens=1000)
#     print(f"Insights and Recommendations for '{parameter}':\n")
#     print(insights)
#     return insights




############ multi-modal code #######################


# def build_unified_index(excel_json_dir: str, other_text_dir: str):
#     """
#     Build a single FAISS index over:
#       - Excel time-series categories (one doc per category)
#       - Other text files (one doc per file)
#     Returns (index, documents, embedder)
#     where documents is a list of dicts with metadata.
#     """
#     documents = []
#     # 1) Excel JSON entries
#     for root, _, files in os.walk(excel_json_dir):
#         for fname in files:
#             if not fname.endswith('.json'): continue
#             path = os.path.join(root, fname)
#             content = json.load(open(path, 'r', encoding='utf-8'))
#             meta = content.get("metadata", {})
#             data = content.get("data", {})
#             for category, series in data.items():
#                 # Represent time-series doc by its category name + a summary
#                 summary = f"Category: {category}. Source: {meta.get('file_name')} / {meta.get('sheet_name')}."
#                 documents.append({
#                     "id": len(documents),
#                     "type": "timeseries",
#                     "text": summary,
#                     "category": category,
#                     "file": meta.get("file_name"),
#                     "sheet": meta.get("sheet_name"),
#                     "series": series
#                 })

#     # 2) Other text files
#     for root, _, files in os.walk(other_text_dir):
#         for fname in files:
#             if not fname.lower().endswith('.txt'): continue
#             path = os.path.join(root, fname)
#             text = open(path, 'r', encoding='utf-8').read()
#             documents.append({
#                 "id": len(documents),
#                 "type": "text",
#                 "text": text,
#                 "file": fname
#             })

#     # 3) Embed
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
#     texts = [doc["text"] for doc in documents]
#     embeddings = embedder.encode(texts, convert_to_numpy=True)

#     # 4) Build FAISS index
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     print(f"Unified FAISS index built with {index.ntotal} documents.")
#     return index, documents, embedder

# def retrieve_relevant_docs(query: str, index, documents, embedder, top_k: int = 5):
#     """
#     Returns the top_k document dicts most similar to the query.
#     """
#     q_emb = embedder.encode([query], convert_to_numpy=True)
#     distances, indices = index.search(q_emb, top_k)
#     return [documents[i] for i in indices[0]]

def build_unified_index(excel_json_dir: str, other_text_dir: str):
    """
    Build a single FAISS index over:
      1) Excel time‑series (one doc per category, with metadata & raw series)
      2) Other extracted text files (one doc per .txt)
    Returns: (index, documents, embedder)
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    documents = []

    # 1) Excel JSON entries → timeseries docs
    for fp in find_json_files(excel_json_dir):
        with open(fp, 'r', encoding='utf-8') as f:
            content = json.load(f)
        meta = content.get("metadata", {})
        for cat, series in content.get("data", {}).items():
            summary = f"Category: {cat}. Source: {meta.get('file_name')} / {meta.get('sheet_name')}."
            documents.append({
                "type": "timeseries",
                "text": summary,
                "category": cat,
                "file": meta.get("file_name"),
                "sheet": meta.get("sheet_name"),
                "series": series
            })

    # 2) Other text files → text docs
    for fname in os.listdir(other_text_dir):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(other_text_dir, fname)
        text = open(path, 'r', encoding='utf-8').read()
        documents.append({
            "type": "text",
            "text": text,
            "file": fname
        })

    # 3) Embed all documents
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [d["text"] for d in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    # 4) Build the FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"Unified FAISS index built with {index.ntotal} documents.")
    return index, documents, embedder

def retrieve_relevant_docs(query: str, index, documents, embedder, top_k: int = 5):
    """
    Given a user query, return the top_k documents (timeseries & text) most similar.
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    return [documents[i] for i in indices[0]]

# def generate_insights_from_predictions(predicted_values, parameter, generate_response_func):
#     # If predicted_values is a list (raw Excel data), compute forecast first:
#     if isinstance(predicted_values, list):
#         dates = [entry["Date"] for entry in predicted_values]
#         y_values = [entry["value"] for entry in predicted_values]
#         dates = pd.to_datetime(dates)
#         df = pd.DataFrame({'ds': dates, 'y': y_values})
#         model = Prophet()
#         model.fit(df)
#         future = model.make_future_dataframe(periods=14, freq='M')
#         forecast = model.predict(future)
#         last_date = max(dates)
#         forecast_filtered = forecast[forecast['ds'] > last_date]
#         predicted_values_df = forecast_filtered[['ds', 'yhat']]
#     else:
#         predicted_values_df = predicted_values  # Assume it is a DataFrame

#     summary_text = (
#         f"The parameter '{parameter}' has been analyzed with the following predicted values "
#         "for each month from the last observed date onward:\n\n"
#     )
#     summary_text += predicted_values_df.to_string(index=False, header=False)
#     summary_text += (
#         "\n\nProvide insights, recommendations, and suggestions to improve decision-making for this parameter."
#     )
#     system_prompt = (
#         "You are a highly skilled business consultant specializing in data-driven decision-making. "
#         "Analyze the provided predictions and generate actionable insights and recommendations."
#     )
#     full_prompt = system_prompt + "\n\n" + summary_text
#     # Call with one positional arg:
#     insights = generate_response_func(full_prompt)
#     #insights = generate_response_func(full_prompt, temperature=0.7, max_tokens=1000)

#     print(f"Insights and Recommendations for '{parameter}':\n")
#     print(insights)
#     return insights

#####newdata

def build_product_index(json_root: str):
    """
    Walk json_root (recursively) and pull out every "(product, date)" pair
    as a tiny document for FAISS.  Returns (index, documents, embedder),
    where each documents[i] has keys: text, product, file.
    """
    documents = []
    for dirpath, _, files in os.walk(json_root):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            c = json.load(open(path, "r", encoding="utf-8"))
            # 1) stock reports
            if "ReportDate" in c and "Items" in c:
                date = c["ReportDate"]
                for item in c["Items"]:
                    prod = item.get("Product")
                    if prod:
                        txt = f"Product: {prod}. Date: {date}."
                        documents.append({"text": txt, "product": prod, "file": fn})
            # 2) invoices & purchase orders
            elif "OrderDate" in c and "Products" in c:
                date = c["OrderDate"]
                for p in c["Products"]:
                    prod = p.get("ProductName") or p.get("Product")
                    if prod:
                        txt = f"Product: {prod}. Date: {date}."
                        documents.append({"text": txt, "product": prod, "file": fn})
            # 3) shipping orders
            elif "OrderDetails" in c and "Products" in c["OrderDetails"]:
                od = c["OrderDetails"]
                date = od.get("OrderDate")
                for p in od["Products"]:
                    prod = p.get("Product")
                    if prod:
                        txt = f"Product: {prod}. Date: {date}."
                        documents.append({"text": txt, "product": prod, "file": fn})
    # embed + build index
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [d["text"] for d in documents]
    embs  = embedder.encode(texts, convert_to_numpy=True)
    dim = embs.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embs)
    print(f"Product FAISS index built with {idx.ntotal} docs.")
    return idx, documents, embedder

def extract_product_timeseries(json_root, product_name):
    """Extract time series data for a product from all JSON files."""
    print(f"\nExtracting time series for product: {product_name}")
    
    # First, aggregate all product data
    product_data, category_data = aggregate_product_data(json_root)
    
    if product_name not in product_data:
        print(f"Product {product_name} not found in aggregated data")
        return []
    
    print(f"Found product data with {len(product_data[product_name]['sales_data'])} sales records")
    
    # Combine all data points into a single time series
    series = []
    
    # Add sales data
    for point in product_data[product_name]["sales_data"]:
        if isinstance(point, dict):  # Ensure point is a dictionary
            try:
                series.append({
                    "Date": point.get("date", ""),
                    "value": float(point.get("value", 0)),
                    "type": "sales"
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing sales data point: {e}")
    
    # Add order data
    for point in product_data[product_name]["order_data"]:
        if isinstance(point, dict):  # Ensure point is a dictionary
            try:
                series.append({
                    "Date": point.get("date", ""),
                    "value": float(point.get("value", 0)),
                    "type": "order"
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing order data point: {e}")
    
    # Add inventory data
    for point in product_data[product_name]["inventory_data"]:
        if isinstance(point, dict):  # Ensure point is a dictionary
            try:
                series.append({
                    "Date": point.get("date", ""),
                    "value": float(point.get("value", 0)),
                    "type": "inventory"
                })
            except (ValueError, TypeError) as e:
                print(f"Error processing inventory data point: {e}")
    
    print(f"Total data points collected: {len(series)}")
    
    # Convert to DataFrame and aggregate by month
    if series:
        try:
            df = pd.DataFrame(series)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.groupby([df["Date"].dt.strftime("%Y-%m"), "type"])["value"].sum().reset_index()
            df["Date"] = pd.to_datetime(df["Date"] + "-01")  # Convert back to datetime with first day of month
            print(f"Successfully aggregated data into {len(df)} monthly records")
            return df.to_dict("records")
        except Exception as e:
            print(f"Error aggregating data: {e}")
            return []
    
    return []

# def generate_insights_from_predictions(predicted_values, parameter, generate_response_func):
#     # If they already passed a DataFrame, normalize its column names:
#     if isinstance(predicted_values, pd.DataFrame):
#         df = predicted_values.copy()
#         if 'Date' in df.columns:
#             df = df.rename(columns={'Date':'ds'})
#         if 'Predicted Value' in df.columns:
#             df = df.rename(columns={'Predicted Value':'yhat'})
#         predicted_df = df[['ds','yhat']]
#     else:
#         # They passed a list of raw entries → run Prophet internally
#         dates = [e["Date"] for e in predicted_values]
#         y_values = [e["value"] for e in predicted_values]
#         dates = pd.to_datetime(dates)
#         df2 = pd.DataFrame({'ds': dates, 'y': y_values})
#         m = Prophet(); m.fit(df2)
#         future = m.make_future_dataframe(periods=14, freq='M')
#         fc = m.predict(future)
#         last = df2['ds'].max()
#         predicted_df = fc[fc['ds'] > last][['ds','yhat']]

#     # Build the prompt
#     summary = "\n".join(f"{row['ds'].date()}: {row['yhat']:.2f}"
#                         for _, row in predicted_df.iterrows())
#     system_prompt = (
#         "You are a seasoned business consultant. "
#         "Here are the forecasted values:"
#     )
#     user_prompt = summary + "\n\n"
#     user_prompt += "Provide actionable insights, recommendations, and suggestions."
    
#     # TWO-argument call
#     return generate_response_func(system_prompt, user_prompt)



def generate_product_graphs(data):
    """Generate comprehensive graphs for product or category data, with forecasts as dotted lines and no negative forecasted values, using highly distinct colors for each legend."""
    if not data:
        print("No data provided for graph generation")
        return {}
    
    import numpy as np
    try:
        print("\nGenerating graphs with data:")
        print(f"Sales data points: {len(data.get('sales_data', []))}")
        print(f"Inventory data points: {len(data.get('inventory_data', []))}")
        print(f"Order data points: {len(data.get('order_data', []))}")
        
        # Highly distinct color palette
        distinct_colors = [
            '#e41a1c', # Red
            '#377eb8', # Blue
            '#4daf4a', # Green
            '#ffeb3b', # Yellow
            '#f781bf', # Pink
            '#ff7f00', # Orange
            '#a65628', # Brown
            '#984ea3', # Purple
            '#000000', # Black
            '#999999', # Gray
        ]
        # Map data types and variants to colors
        type_color_map = {}
        all_types = ["sales", "inventory", "order"]
        for idx, t in enumerate(all_types):
            type_color_map[t] = distinct_colors[idx % len(distinct_colors)]
        # Assign unique colors for moving averages and forecast variants
        ma7_color = distinct_colors[3]  # Yellow
        ma30_color = distinct_colors[4] # Pink
        forecast_color = distinct_colors[7] # Purple
        upper_color = distinct_colors[0] # Red
        lower_color = distinct_colors[1] # Blue
        # Convert data to DataFrame
        df = pd.DataFrame()
        for data_type in ["sales_data", "inventory_data", "order_data"]:
            if data.get(data_type):
                temp_df = pd.DataFrame(data[data_type])
                temp_df["type"] = data_type.replace("_data", "")
                df = pd.concat([df, temp_df])
        
        if df.empty:
            print("No data points found after conversion to DataFrame")
            return {}
        
        print(f"\nTotal data points: {len(df)}")
        
        # Handle date parsing with multiple formats
        def parse_date(date_str):
            try:
                for fmt in ['%Y-%m-%d', '%Y-%m', '%Y/%m/%d', '%Y/%m']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                return pd.to_datetime(date_str)
            except:
                print(f"Warning: Could not parse date: {date_str}")
                return None
        
        df["date"] = df["date"].apply(parse_date)
        df = df.dropna(subset=["date"])  # Remove rows with invalid dates
        df = df.sort_values("date")
        
        print(f"Valid data points after date parsing: {len(df)}")
        
        # Calculate various metrics
        monthly_data = df.groupby([df["date"].dt.strftime("%Y-%m"), "type"])["value"].agg(['sum', 'mean', 'count']).reset_index()
        monthly_data["date"] = pd.to_datetime(monthly_data["date"] + "-01")
        
        # Calculate moving averages
        df['MA7'] = df.groupby('type')['value'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df['MA30'] = df.groupby('type')['value'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
        
        # Generate Prophet forecasts for each type
        forecasts = {}
        for data_type in df["type"].unique():
            type_df = df[df["type"] == data_type].copy()
            if len(type_df) >= 2:  # Prophet requires at least 2 data points
                type_df = type_df.rename(columns={'date': 'ds', 'value': 'y'})
                try:
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                    model.fit(type_df[['ds', 'y']])
                    future = model.make_future_dataframe(periods=12, freq='M')
                    forecast = model.predict(future)
                    # CLIP negative values to zero
                    for col in ['yhat', 'yhat_upper', 'yhat_lower']:
                        forecast[col] = forecast[col].clip(lower=0)
                    forecasts[data_type] = forecast
                    print(f"Generated forecast for {data_type} with {len(forecast)} points")
                except Exception as e:
                    print(f"Warning: Could not generate forecast for {data_type}: {str(e)}")
        
        # Create the plots dictionary
        plots = {}
        
        # Time Series Plot (with forecast overlay)
        time_series_data = []
        for idx, data_type in enumerate(df["type"].unique()):
            color = type_color_map.get(data_type, distinct_colors[(idx+2)%len(distinct_colors)])
            type_df = df[df["type"] == data_type]
            time_series_data.append({
                'x': type_df["date"].dt.strftime("%Y-%m-%d").tolist(),
                'y': type_df["value"].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{data_type.title()} (Observed)',
                'line': {'width': 2, 'color': color},
                'marker': {'size': 6}
            })
            # Add forecast as dotted line if available
            if data_type in forecasts:
                forecast = forecasts[data_type]
                last_obs_date = type_df["date"].max()
                forecast_future = forecast[forecast['ds'] > last_obs_date]
                if not forecast_future.empty:
                    time_series_data.append({
                        'x': forecast_future['ds'].dt.strftime("%Y-%m-%d").tolist(),
                        'y': forecast_future['yhat'].tolist(),
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': f'{data_type.title()} (Forecast)',
                        'line': {'width': 2, 'color': forecast_color, 'dash': 'dot'}
                    })
        plots['time_series'] = {
            'title': 'Time Series Analysis',
            'x_label': 'Date',
            'y_label': 'Value',
            'data': time_series_data
        }
        
        # Monthly Analysis Plot
        monthly_data_list = []
        for idx, data_type in enumerate(monthly_data["type"].unique()):
            color = type_color_map.get(data_type, distinct_colors[(idx+3)%len(distinct_colors)])
            type_data = monthly_data[monthly_data["type"] == data_type]
            monthly_data_list.append({
                'x': type_data["date"].dt.strftime("%Y-%m").tolist(),
                'y': type_data["sum"].tolist(),
                'type': 'bar',
                'name': f'{data_type.title()} Total',
                'marker': {'color': color}
            })
            monthly_data_list.append({
                'x': type_data["date"].dt.strftime("%Y-%m").tolist(),
                'y': type_data["mean"].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': f'{data_type.title()} Average',
                'line': {'width': 2, 'color': distinct_colors[(idx+5)%len(distinct_colors)]}
            })
        plots['monthly_analysis'] = {
            'title': 'Monthly Analysis',
            'x_label': 'Month',
            'y_label': 'Value',
            'data': monthly_data_list
        }
        
        # Moving Averages Plot
        ma_data = []
        for idx, data_type in enumerate(df["type"].unique()):
            type_df = df[df["type"] == data_type]
            ma_data.append({
                'x': type_df["date"].dt.strftime("%Y-%m-%d").tolist(),
                'y': type_df["MA7"].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': f'{data_type.title()} 7-day MA',
                'line': {'width': 2, 'color': ma7_color}
            })
            ma_data.append({
                'x': type_df["date"].dt.strftime("%Y-%m-%d").tolist(),
                'y': type_df["MA30"].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': f'{data_type.title()} 30-day MA',
                'line': {'width': 2, 'color': ma30_color}
            })
        plots['moving_averages'] = {
            'title': 'Moving Averages',
            'x_label': 'Date',
            'y_label': 'Value',
            'data': ma_data
        }
        
        # Forecast Plot (future only, dotted)
        if forecasts:
            forecast_data = []
            for idx, (data_type, forecast) in enumerate(forecasts.items()):
                type_df = df[df["type"] == data_type]
                last_obs_date = type_df["date"].max()
                forecast_future = forecast[forecast['ds'] > last_obs_date]
                if not forecast_future.empty:
                    forecast_data.append({
                        'x': forecast_future['ds'].dt.strftime("%Y-%m-%d").tolist(),
                        'y': forecast_future['yhat'].tolist(),
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': f'{data_type.title()} Forecast',
                        'line': {'width': 2, 'color': forecast_color, 'dash': 'dot'}
                    })
                    forecast_data.append({
                        'x': forecast_future['ds'].dt.strftime("%Y-%m-%d").tolist(),
                        'y': forecast_future['yhat_upper'].tolist(),
                        'type': 'scatter',
                        'mode': 'lines',
                        'line': {'dash': 'dash', 'color': upper_color},
                        'name': f'{data_type.title()} Upper Bound'
                    })
                    forecast_data.append({
                        'x': forecast_future['ds'].dt.strftime("%Y-%m-%d").tolist(),
                        'y': forecast_future['yhat_lower'].tolist(),
                        'type': 'scatter',
                        'mode': 'lines',
                        'line': {'dash': 'dash', 'color': lower_color},
                        'name': f'{data_type.title()} Lower Bound'
                    })
            plots['forecast'] = {
                'title': 'Forecast Analysis',
                'x_label': 'Date',
                'y_label': 'Value',
                'data': forecast_data
            }
        print(f"\nGenerated {len(plots)} plot types:")
        for plot_type, plot_data in plots.items():
            print(f"- {plot_type}: {len(plot_data['data'])} data series")
            for series in plot_data['data']:
                print(f"  Series: {series['name']}")
                print(f"  Data points: {len(series['x'])}")
        return plots
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return {}

def generate_insights_from_predictions(data):
    """Generate actionable, detailed, and user-friendly insights, recommendations, and alerts from the data."""
    try:
        insights = []
        # Helper for date parsing
        def parse_date(date_str):
            try:
                for fmt in ['%Y-%m-%d', '%Y-%m', '%Y/%m/%d', '%Y/%m']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                return pd.to_datetime(date_str)
            except:
                return None
        # Sales Analysis
        if "sales_data" in data:
            sales_df = pd.DataFrame(data["sales_data"])
            sales_df["date"] = sales_df["date"].apply(parse_date)
            sales_df = sales_df.dropna(subset=["date"])
            if not sales_df.empty:
                total = sales_df["value"].sum()
                avg = sales_df["value"].mean()
                mx = sales_df["value"].max()
                mn = sales_df["value"].min()
                std = sales_df["value"].std()
                insights.append(f"<b>Sales Analysis:</b> Total: {total:.2f}, Average: {avg:.2f}, Max: {mx:.2f}, Min: {mn:.2f}, Std: {std:.2f}")
                # Trend
                if len(sales_df) > 1:
                    growth = ((sales_df["value"].iloc[-1] - sales_df["value"].iloc[0]) / abs(sales_df["value"].iloc[0]+1e-9)) * 100
                    if growth > 10:
                        insights.append("<span style='color:green'><b>Sales are increasing.</b></span> Consider increasing inventory or running promotions to capitalize on growth.")
                    elif growth < -10:
                        insights.append("<span style='color:red'><b>Sales are declining.</b></span> Investigate causes (seasonality, competition, stockouts) and consider targeted marketing or pricing adjustments.")
                    else:
                        insights.append("<b>Sales are stable.</b> Maintain current strategy but monitor for changes.")
        # Inventory Analysis
        if "inventory_data" in data:
            inv_df = pd.DataFrame(data["inventory_data"])
            inv_df["date"] = inv_df["date"].apply(parse_date)
            inv_df = inv_df.dropna(subset=["date"])
            if not inv_df.empty:
                total = inv_df["value"].sum()
                avg = inv_df["value"].mean()
                mx = inv_df["value"].max()
                mn = inv_df["value"].min()
                std = inv_df["value"].std()
                insights.append(f"<b>Inventory Analysis:</b> Total: {total:.2f}, Average: {avg:.2f}, Max: {mx:.2f}, Min: {mn:.2f}, Std: {std:.2f}")
                # Alert for low inventory
                if avg < 10 or mn < 5:
                    insights.append("<span style='color:red'><b>Warning: Inventory is low.</b></span> Consider restocking soon to avoid stockouts.")
                elif avg > 100:
                    insights.append("<span style='color:orange'><b>Inventory is high.</b></span> Consider promotions or discounts to reduce excess stock.")
        # Order Analysis
        if "order_data" in data:
            order_df = pd.DataFrame(data["order_data"])
            order_df["date"] = order_df["date"].apply(parse_date)
            order_df = order_df.dropna(subset=["date"])
            if not order_df.empty:
                total = order_df["value"].sum()
                avg = order_df["value"].mean()
                mx = order_df["value"].max()
                mn = order_df["value"].min()
                std = order_df["value"].std()
                insights.append(f"<b>Order Analysis:</b> Total: {total:.2f}, Average: {avg:.2f}, Max: {mx:.2f}, Min: {mn:.2f}, Std: {std:.2f}")
        # Forecast-based Recommendations
        try:
            from prophet import Prophet
            for key in ["sales_data", "inventory_data", "order_data"]:
                if key in data and len(data[key]) > 2:
                    df = pd.DataFrame(data[key])
                    df["date"] = df["date"].apply(parse_date)
                    df = df.dropna(subset=["date"])
                    df = df.rename(columns={"date": "ds", "value": "y"})
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                    model.fit(df[["ds", "y"]])
                    future = model.make_future_dataframe(periods=6, freq='M')
                    forecast = model.predict(future)
                    last_obs = df["ds"].max()
                    forecast_future = forecast[forecast['ds'] > last_obs]
                    if not forecast_future.empty:
                        next_val = forecast_future['yhat'].iloc[0]
                        if key == "sales_data":
                            if next_val > df["y"].mean() * 1.1:
                                insights.append("<span style='color:green'><b>Sales are forecasted to increase.</b></span> Prepare for higher demand by increasing inventory or staffing.")
                            elif next_val < df["y"].mean() * 0.9:
                                insights.append("<span style='color:red'><b>Sales are forecasted to decrease.</b></span> Consider cost-saving measures or targeted promotions.")
                        if key == "inventory_data":
                            if next_val < 10:
                                insights.append("<span style='color:red'><b>Inventory is forecasted to be very low soon.</b></span> Immediate restocking is recommended.")
                            elif next_val > 100:
                                insights.append("<span style='color:orange'><b>Inventory is forecasted to be high.</b></span> Plan for clearance sales or discounts.")
                        if key == "order_data":
                            if next_val > df["y"].mean() * 1.1:
                                insights.append("<span style='color:green'><b>Order volume is forecasted to increase.</b></span> Ensure supply chain readiness.")
                            elif next_val < df["y"].mean() * 0.9:
                                insights.append("<span style='color:red'><b>Order volume is forecasted to decrease.</b></span> Investigate possible causes.")
        except Exception as e:
            insights.append(f"<span style='color:gray'>Could not generate forecast-based recommendations: {str(e)}</span>")
        # Opportunities and Alerts
        if "sales_data" in data and "inventory_data" in data:
            sales_df = pd.DataFrame(data["sales_data"])
            inv_df = pd.DataFrame(data["inventory_data"])
            sales_df["date"] = sales_df["date"].apply(parse_date)
            inv_df["date"] = inv_df["date"].apply(parse_date)
            sales_df = sales_df.dropna(subset=["date"])
            inv_df = inv_df.dropna(subset=["date"])
            if not sales_df.empty and not inv_df.empty:
                if sales_df["value"].iloc[-1] > inv_df["value"].iloc[-1]:
                    insights.append("<span style='color:red'><b>Warning: Sales are outpacing inventory.</b></span> Risk of stockout. Consider urgent restocking.")
                elif inv_df["value"].iloc[-1] > sales_df["value"].iloc[-1] * 2:
                    insights.append("<span style='color:orange'><b>Inventory is much higher than sales.</b></span> Consider promotions to reduce excess stock.")
        # Sales and Order correlation
        if "sales_data" in data and "order_data" in data:
            sales_df = pd.DataFrame(data["sales_data"])
            order_df = pd.DataFrame(data["order_data"])
            sales_df["date"] = sales_df["date"].apply(parse_date)
            order_df["date"] = order_df["date"].apply(parse_date)
            sales_df = sales_df.dropna(subset=["date"])
            order_df = order_df.dropna(subset=["date"])
            if not sales_df.empty and not order_df.empty:
                monthly_sales = sales_df.groupby(sales_df["date"].dt.strftime("%Y-%m"))["value"].sum()
                monthly_orders = order_df.groupby(order_df["date"].dt.strftime("%Y-%m"))["value"].sum()
                if len(monthly_sales) > 1 and len(monthly_orders) > 1:
                    sales_growth = ((monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / (abs(monthly_sales.iloc[-2])+1e-9)) * 100
                    order_growth = ((monthly_orders.iloc[-1] - monthly_orders.iloc[-2]) / (abs(monthly_orders.iloc[-2])+1e-9)) * 100
                    insights.append(f"<b>Sales and Order Correlation:</b> Sales Growth: {sales_growth:.1f}%, Order Growth: {order_growth:.1f}%")
                    if abs(sales_growth - order_growth) > 10:
                        insights.append("<span style='color:orange'>There's a significant gap between sales and order growth.</span>")
                        if sales_growth > order_growth:
                            insights.append("<b>Consider increasing order quantities to meet demand.</b>")
                        else:
                            insights.append("<b>Consider reducing order quantities to prevent overstock.</b>")
        # General suggestion
        insights.append("<b>General Recommendation:</b> Regularly monitor sales, inventory, and orders. Use forecasts to plan inventory and marketing. Respond quickly to alerts and opportunities.")
        return "<ul>" + "".join([f"<li>{i}</li>" for i in insights]) + "</ul>"
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return f"Error generating insights: {str(e)}"

def aggregate_product_data(json_root):
    """Aggregate product data from all JSON files."""
    print(f"\nAggregating data from: {json_root}")
    product_data = {}
    category_data = {}
    
    def add_to_product_data(product_name, category, date, value, data_type):
        if product_name not in product_data:
            product_data[product_name] = {
                "sales_data": [],
                "inventory_data": [],
                "order_data": [],
                "category": category
            }
        
        if category not in category_data:
            category_data[category] = {
                "sales_data": [],
                "inventory_data": [],
                "order_data": [],
                "products": set()
            }
        
        category_data[category]["products"].add(product_name)
        
        data_point = {
            "date": date,
            "value": value
        }
        
        product_data[product_name][f"{data_type}_data"].append(data_point)
        category_data[category][f"{data_type}_data"].append(data_point)
    
    def process_file(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Determine file type from path
            if "Inventory Report" in file_path:
                # Process inventory report
                report_date = data.get("ReportDate", "")
                print(f"Processing inventory report from {report_date}")
                for item in data.get("Items", []):
                    product = item.get("Product", "")
                    category = item.get("Category", "")
                    units_sold = float(item.get("UnitsSold", 0))
                    units_in_stock = float(item.get("UnitsInStock", 0))
                    
                    if product and category:
                        add_to_product_data(product, category, report_date, units_sold, "sales")
                        add_to_product_data(product, category, report_date, units_in_stock, "inventory")
                        print(f"Added data for product: {product}, category: {category}")
            
            elif "Shipping orders" in file_path:
                # Process shipping orders
                order_date = data.get("OrderDetails", {}).get("OrderDate", "")
                print(f"Processing shipping order from {order_date}")
                for product_info in data.get("OrderDetails", {}).get("Products", []):
                    product = product_info.get("Product", "")
                    quantity = float(product_info.get("Quantity", 0))
                    
                    if product:
                        # Find category from inventory data
                        category = None
                        for cat, cat_data in category_data.items():
                            if product in cat_data["products"]:
                                category = cat
                                break
                        
                        if category:
                            add_to_product_data(product, category, order_date, quantity, "order")
                            print(f"Added order data for product: {product}, category: {category}")
            
            elif "PurchaseOrders" in file_path:
                # Process purchase orders
                order_date = data.get("OrderDate", "")
                print(f"Processing purchase order from {order_date}")
                for product_info in data.get("Products", []):
                    product = product_info.get("Product", "")
                    quantity = float(product_info.get("Quantity", 0))
                    
                    if product:
                        # Find category from inventory data
                        category = None
                        for cat, cat_data in category_data.items():
                            if product in cat_data["products"]:
                                category = cat
                                break
                        
                        if category:
                            add_to_product_data(product, category, order_date, quantity, "order")
                            print(f"Added order data for product: {product}, category: {category}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Get all JSON files
    json_files = []
    for root, _, files in os.walk(json_root):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_file, json_files)
    
    # Sort data points by date for each product and category
    for product in product_data.values():
        for data_type in ["sales_data", "inventory_data", "order_data"]:
            product[data_type].sort(key=lambda x: x["date"])
    
    for category in category_data.values():
        for data_type in ["sales_data", "inventory_data", "order_data"]:
            category[data_type].sort(key=lambda x: x["date"])
        category["products"] = list(category["products"])
    
    # Print summary of aggregated data
    print(f"\nAggregated data summary:")
    print(f"Products found: {len(product_data)}")
    for product_name, data in product_data.items():
        print(f"\nProduct: {product_name}")
        print(f"Category: {data['category']}")
        print(f"Sales data points: {len(data['sales_data'])}")
        print(f"Inventory data points: {len(data['inventory_data'])}")
        print(f"Order data points: {len(data['order_data'])}")
    
    print(f"\nCategories found: {len(category_data)}")
    for category_name, data in category_data.items():
        print(f"\nCategory: {category_name}")
        print(f"Products: {len(data['products'])}")
        print(f"Sales data points: {len(data['sales_data'])}")
        print(f"Inventory data points: {len(data['inventory_data'])}")
        print(f"Order data points: {len(data['order_data'])}")
    
    return product_data, category_data

def save_aggregated_data(product_data, category_data, output_dir):
    """Save the aggregated product and category data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save product data
    products_dir = os.path.join(output_dir, 'products')
    os.makedirs(products_dir, exist_ok=True)
    
    for product_name, data in product_data.items():
        # Create a copy of the data to avoid modifying the original
        save_data = {
            "sales_data": data["sales_data"],
            "inventory_data": data["inventory_data"],
            "order_data": data["order_data"],
            "category": data["category"]
        }
        
        # Save to file
        safe_product_name = "".join(c if c.isalnum() else "_" for c in product_name)
        output_file = os.path.join(products_dir, f"{safe_product_name}.json")
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    # Save category data
    categories_dir = os.path.join(output_dir, 'categories')
    os.makedirs(categories_dir, exist_ok=True)
    
    for category_name, data in category_data.items():
        # Create a copy of the data to avoid modifying the original
        save_data = {
            "sales_data": data["sales_data"],
            "inventory_data": data["inventory_data"],
            "order_data": data["order_data"],
            "products": data["products"]  # This is already a list
        }
        
        # Save to file
        safe_category_name = "".join(c if c.isalnum() else "_" for c in category_name)
        output_file = os.path.join(categories_dir, f"{safe_category_name}.json")
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
