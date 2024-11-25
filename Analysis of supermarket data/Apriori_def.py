import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pandas.plotting import parallel_coordinates
import copy
from fuzzywuzzy import process
import networkx as nx


def read_data(file_path: str, sheet_name: str = None, handle_duplicates: bool = True):
    """
    Read data from a file and return a DataFrame. Supports CSV, TXT, Excel, JSON, and HTML files.
    
    Parameters:
    - file_path: The path to the data file.
    - sheet_name: The name of the sheet to read from an Excel file (default is None).
    - handle_duplicates: Whether to drop duplicate rows (default is True).
    
    Returns:
    - A DataFrame or a list of DataFrames (in case of HTML).
    
    Raises:
    - ValueError: If the file format is not supported.
    """
    
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension in ['csv', 'txt']:
            data = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            if sheet_name is None:
                sheet_name = input('Enter the sheet name: ')
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension == 'json':
            data = pd.read_json(file_path)
        elif file_extension == 'html':
            data = pd.read_html(file_path)
            if len(data) == 1:
                data = data[0]
        else:
            raise ValueError('Unsupported file format.')
        
        # Deep copy the data to avoid modifying the original data
        df = copy.deepcopy(data)
        
        # Handle duplicates if required
        if handle_duplicates:
            duplicated_num = df.duplicated().sum()
            if duplicated_num == 0:
                print('the DataFrame dont have any duplicates row')
            else:
                print(f'the DataFrame have {duplicated_num} duplicates rows')
                df = df.drop_duplicates()
                print('the DataFrame without duplicates rows')
        
        print(f'Data read successfully from {file_path}')
        return df
    
    except Exception as e:
        print(f'Error reading data from {file_path}: {str(e)}')
        raise


def columns_info(df):
    cols=[]
    dtype=[]
    unique_v=[]
    n_unique_v=[]
    number_of_rows = df.shape[0]
    number_of_null = []
    for col in df.columns:
        cols.append(col)
        dtype.append(df[col].dtypes)
        unique_v.append(df[col].unique())
        n_unique_v.append(df[col].nunique())
        number_of_null.append(df[col].isnull().sum())
    
    return pd.DataFrame({'names':cols, 'dtypes':dtype, 'unique':unique_v, 'n_unique':n_unique_v, 'number_of_rows':number_of_rows, 'number_of_null':number_of_null})


def bar_plot(df,col):
    
    fig = px.bar(df,
        x = df[col].value_counts().keys(), 
        y = df[col].value_counts().values,
        color= df[col].value_counts().keys())
    fig.update_layout(
    xaxis_title= col,
    yaxis_title="Count",
    legend_title=col,
    font_family="Courier New",
    font_color="blue",
    title_font_family="Times New Roman",
    title_font_color="red",
    legend_title_font_color="green")
    fig.show()


def transaction_to_df(transactions):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_ary, columns = te.columns_)
    return te_df


def plot_item_frequency(df):
    item_frequency = df.sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    item_frequency[:20].plot(kind='bar')
    plt.title('Top 20 Most Frequent Items')
    plt.xlabel('Items')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def association_rules_apriori(te_df, min_support = 0.001, min_threshold = 0.001, metric = 'lift', use_colnames = True, verbose = 1):
    freq_items = apriori(te_df, min_support = min_support, use_colnames = use_colnames, verbose = verbose)
    freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
    rules =  association_rules(freq_items,metric = metric,min_threshold = min_threshold)
    return freq_items, rules


def get_top_related_products(rules_df, product_name, n=5, similarity_threshold=75, plot_output=False):
    """
    Get top N products that are frequently sold with the given product.
    If the exact product is not found, prompts the user to choose a similar product or exits gracefully.
    Optionally, plots the top 2*N related products.
    
    Parameters:
    -----------
    rules_df : pandas DataFrame
        Association rules DataFrame containing the rules
    product_name : str
        Name of the product to find associations for
    n : int, optional (default=5)
        Number of top products to return
    similarity_threshold : int, optional (default=80)
        Minimum similarity score to consider a product name match
    plot_output : bool, optional (default=False)
        If True, generates a plot of the top 2*N related products
    
    Returns:
    --------
    None
        Prints the results directly or exits if no valid product name is found.
    """
    # Get all unique products from rules
    all_products = set()
    for items in set(rules_df['antecedents']).union(set(rules_df['consequents'])):
        all_products.update(items)
    
    # Check if product exists
    while product_name not in all_products:
        # Find similar product names
        similar_products = process.extract(product_name, all_products, limit=5)
        similar_products = [(name, score) for name, score in similar_products if score >= similarity_threshold]
        
        if similar_products:
            print(f"\nProduct '{product_name}' not found. Did you mean one of these?")
            for idx, (name, score) in enumerate(similar_products, start=1):
                print(f"{idx}. {name} (similarity: {score}%)")
            print("Enter the number of your choice or type the product name directly.")
            
            user_input = input("Your choice: ").strip()
            
            # Handle user input
            if user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(similar_products):
                    product_name = similar_products[choice - 1][0]
                else:
                    print("Invalid choice. Please try again.")
                    continue
            elif user_input in all_products:
                product_name = user_input
            else:
                print("Invalid product name. Exiting.")
                return
        else:
            print(f"Product '{product_name}' not found and no similar products are available.")
            return
    
    # Original functionality for exact matches
    product = frozenset([product_name])
    
    # Filter rules where the product appears in antecedents or consequents
    mask1 = rules_df['antecedents'].apply(lambda x: product.issubset(x))
    mask2 = rules_df['consequents'].apply(lambda x: product.issubset(x))
    
    related_rules = pd.concat([
        rules_df[mask1][['antecedents', 'consequents', 'lift', 'confidence']],
        rules_df[mask2][['consequents', 'antecedents', 'lift', 'confidence']]
    ])
    
    related_rules['related_product'] = related_rules.apply(
        lambda row: list(row['consequents'] - product if product.issubset(row['antecedents']) 
                        else row['antecedents'] - product)[0]
        if len(row['consequents']) == 1 or len(row['antecedents']) == 1
        else None, axis=1
    )
    
    related_rules = related_rules.dropna(subset=['related_product']).drop_duplicates(subset=['related_product'])
    top_related = related_rules.nlargest(2 * n, 'lift')[['related_product', 'lift', 'confidence']]
    
    # Print results
    print(f"\nTop {n} products frequently bought with '{product_name}':")
    print(top_related.head(n))
    
    # Generate plot if requested
    if plot_output:
        # Limit plot to 2*N products
        plot_data = top_related.head(2 * n)
        plt.figure(figsize=(10, 6))
        plt.barh(plot_data['related_product'], plot_data['lift'], color='purple', edgecolor='black')
        plt.xlabel('Lift')
        plt.ylabel('Related Products')
        plt.title(f"Top {2 * n} Related Products for '{product_name}'")
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.tight_layout()
        plt.show()


def plot_enhanced_product_network(rules, min_lift=1.5, min_confidence=0.05):
    # Create network graph
    G = nx.Graph()
    
    # Filter rules based on both lift and confidence
    filtered_rules = rules[(rules['lift'] >= min_lift) & 
                          (rules['confidence'] >= min_confidence)]
    
    # Add edges for both antecedents and consequents
    for idx, row in filtered_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for ant in antecedents:
            for cons in consequents:
                G.add_node(ant, support=row['antecedent support'])
                G.add_node(cons, support=row['consequent support'])
                G.add_edge(ant, cons, 
                          weight=row['lift'],
                          confidence=row['confidence'])
    
    # Calculate node sizes and edge widths
    node_sizes = [G.nodes[node]['support'] * 5000 for node in G.nodes()]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=node_sizes,
                          alpha=0.7,
                          ax=ax)
    
    nx.draw_networkx_edges(G, pos,
                          width=edge_widths,
                          alpha=0.5,
                          edge_color='gray',
                          ax=ax)
    
    nx.draw_networkx_labels(G, pos,
                           font_size=8,
                           font_weight='bold',
                           ax=ax)
    
    # Add title
    ax.set_title('Product Relationship Network\n'
                f'(Min Lift: {min_lift}, Min Confidence: {min_confidence})',
                fontsize=16, pad=20)
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                              norm=plt.Normalize(vmin=min(node_sizes),
                                               vmax=max(node_sizes)))
    plt.colorbar(sm, ax=ax, label='Node Size (Support)')
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print network statistics
    print(f"Network Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    
    # Return top 5 products by degree centrality
    degree_cent = nx.degree_centrality(G)
    top_products = sorted(degree_cent.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:5]
    print("\nTop 5 Products by Connections:")
    for product, centrality in top_products:
        print(f"{product}: {centrality:.3f}")


def plot_rule_px_scatter(rules, x = 'support', y = 'confidence'):
    fig=px.scatter(rules[x], rules[y])
    fig.update_layout(
        xaxis_title="support",
        yaxis_title="confidence",
    
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        title=(f'{x} vs {y}')
        
    )
    fig.show()


def plot_rule_scatter(rules, x = 'support', y = 'confidence'):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules[x], rules[y], alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y} Scatter Plot')
    plt.tight_layout()
    plt.show()


def plot_rule_polyfit(rules, x = 'lift', y = 'confidence'):
    fit = np.polyfit(rules[x], rules[y], 1)
    fit_fn = np.poly1d(fit)
    plt.plot(rules[x], rules[y], 'yo', rules[x], 
    fit_fn(rules[x]))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')




























