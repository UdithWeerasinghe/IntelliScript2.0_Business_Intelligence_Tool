import os, datetime
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import json
from flask import Flask, request, jsonify, send_from_directory
from src import analysis, llm
from src.agent_graph import run_agentic_pipeline

def create_app():
    # Create Flask app with static folder configuration
    app = Flask(__name__, 
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend'),
        static_url_path='')

    cfg = llm.load_config("config/config.json")

    # Initialize the product index
    json_root = "data/json_input"
    print(f"Loading data from: {json_root}")
    
    # List all JSON files to verify data access
    import glob
    json_files = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        print(f"- {file}")

    index, documents, embedder = analysis.build_product_index(json_root)

    # Create output directory for aggregated data
    aggregated_data_dir = "data/aggregated_products"
    os.makedirs(aggregated_data_dir, exist_ok=True)

    # Aggregate all product data
    product_data, category_data = analysis.aggregate_product_data(json_root)
    print(f"\nAggregated data summary:")
    print(f"Products found: {len(product_data)}")
    print("Product names:", list(product_data.keys()))
    print(f"Categories found: {len(category_data)}")
    print("Category names:", list(category_data.keys()))
    
    analysis.save_aggregated_data(product_data, category_data, aggregated_data_dir)

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/<path:path>')
    def static_proxy(path):
        return app.send_static_file(path)

    @app.route('/analyze', methods=['POST'])
    def analyze():
        try:
            data = request.get_json()
            query = data.get('query', '').lower()
            print(f"\nProcessing query: {query}")
            
            # Aggregate product and category data
            product_data, category_data = analysis.aggregate_product_data('data/json_input')
            print(f"Found {len(product_data)} products and {len(category_data)} categories")
            
            # First try to find an exact product match
            found_product = None
            found_category = None
            
            for product_name in product_data.keys():
                if product_name.lower() in query:
                    found_product = product_name
                    print(f"Found product match: {found_product}")
                    break
            
            # If no product found, try to find a category match
            if not found_product:
                for category_name in category_data.keys():
                    if category_name.lower() in query:
                        found_category = category_name
                        print(f"Found category match: {found_category}")
                        break
            
            if found_product:
                # Generate plots and insights for the product
                plots = analysis.generate_product_graphs(product_data[found_product])
                insights = analysis.generate_insights_from_predictions(product_data[found_product])
                
                print(f"\nGenerated plots for product {found_product}:")
                print(f"Number of plot types: {len(plots)}")
                for plot_type, plot_data in plots.items():
                    print(f"- {plot_type}:")
                    print(f"  Title: {plot_data['title']}")
                    print(f"  Number of data series: {len(plot_data['data'])}")
                    for series in plot_data['data']:
                        print(f"  Series: {series['name']}")
                        print(f"  Data points: {len(series['x'])}")
                
                response_data = {
                    'status': 'success',
                    'type': 'product',
                    'name': found_product,
                    'plots': plots,
                    'insights': insights
                }
                print("\nSending response with plots and insights")
                return jsonify(response_data)
            
            elif found_category:
                # Generate plots and insights for the category
                plots = analysis.generate_product_graphs(category_data[found_category])
                insights = analysis.generate_insights_from_predictions(category_data[found_category])
                
                print(f"\nGenerated plots for category {found_category}:")
                print(f"Number of plot types: {len(plots)}")
                for plot_type, plot_data in plots.items():
                    print(f"- {plot_type}:")
                    print(f"  Title: {plot_data['title']}")
                    print(f"  Number of data series: {len(plot_data['data'])}")
                    for series in plot_data['data']:
                        print(f"  Series: {series['name']}")
                        print(f"  Data points: {len(series['x'])}")
                
                response_data = {
                    'status': 'success',
                    'type': 'category',
                    'name': found_category,
                    'plots': plots,
                    'insights': insights
                }
                print("\nSending response with plots and insights")
                return jsonify(response_data)
            
            else:
                print("No matching product or category found")
                return jsonify({
                    'status': 'error',
                    'message': 'No matching product or category found'
                }), 404
            
        except Exception as e:
            print(f"Error in analyze endpoint: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing request: {str(e)}'
            }), 500

    @app.route('/analyze_agent', methods=['POST'])
    def analyze_agent():
        try:
            data = request.get_json()
            query = data.get('query', '').lower()
            
            # Aggregate product and category data
            product_data, category_data = analysis.aggregate_product_data('data/json_input')
            
            # First try to find an exact product match
            found_product = None
            found_category = None
            
            for product_name in product_data.keys():
                if product_name.lower() in query:
                    found_product = product_name
                    break
            
            # If no product found, try to find a category match
            if not found_product:
                for category_name in category_data.keys():
                    if category_name.lower() in query:
                        found_category = category_name
                        break
            
            if found_product:
                # Generate plots and insights for the product
                plots = analysis.generate_product_graphs(product_data[found_product])
                insights = analysis.generate_insights_from_predictions(product_data[found_product])
                
                return jsonify({
                    'status': 'success',
                    'type': 'product',
                    'name': found_product,
                    'plots': plots,
                    'insights': insights
                })
            
            elif found_category:
                # Generate plots and insights for the category
                plots = analysis.generate_product_graphs(category_data[found_category])
                insights = analysis.generate_insights_from_predictions(category_data[found_category])
                
                return jsonify({
                    'status': 'success',
                    'type': 'category',
                    'name': found_category,
                    'plots': plots,
                    'insights': insights
                })
            
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No matching product or category found'
                }), 404
            
        except Exception as e:
            print(f"Error in analyze_agent endpoint: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing request: {str(e)}'
            }), 500

    return app

# Create the Flask app instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
