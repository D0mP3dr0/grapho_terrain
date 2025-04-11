"""
Functions for visualizing geospatial networks and graphs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from ..core.graph import GeoGraph, MultiLayerGeoGraph


def plot_geograph(graph, figsize=(12, 10), node_size=50, edge_width=1, node_color='blue', 
                 edge_color='gray', node_attr=None, edge_attr=None, node_cmap='viridis', 
                 edge_cmap='viridis', add_basemap=False, title=None, ax=None):
    """
    Plot a GeoGraph with nodes and edges.
    
    Parameters
    ----------
    graph : GeoGraph
        Graph to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    node_size : int or float, optional
        Size of nodes
    edge_width : int or float, optional
        Width of edges
    node_color : str or list, optional
        Color of nodes
    edge_color : str or list, optional
        Color of edges
    node_attr : str, optional
        Node attribute to use for coloring
    edge_attr : str, optional
        Edge attribute to use for coloring
    node_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for nodes
    edge_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for edges
    add_basemap : bool, optional
        Whether to add a basemap
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    # Convert graph to GeoDataFrames
    nodes_gdf, edges_gdf = graph.to_geodataframes()
    
    # Initialize plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot edges
    if edge_attr is not None and edge_attr in edges_gdf.columns:
        edges_gdf.plot(ax=ax, column=edge_attr, cmap=edge_cmap, 
                      linewidth=edge_width, alpha=0.7)
        
        # Add colorbar for edges
        norm = Normalize(vmin=edges_gdf[edge_attr].min(), vmax=edges_gdf[edge_attr].max())
        sm = ScalarMappable(norm=norm, cmap=edge_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, label=edge_attr)
    else:
        edges_gdf.plot(ax=ax, color=edge_color, linewidth=edge_width, alpha=0.7)
    
    # Plot nodes
    if node_attr is not None and node_attr in nodes_gdf.columns:
        nodes_gdf.plot(ax=ax, column=node_attr, cmap=node_cmap, 
                      markersize=node_size, alpha=0.8)
        
        # Add colorbar for nodes (only if not already added for edges)
        if edge_attr is None:
            norm = Normalize(vmin=nodes_gdf[node_attr].min(), vmax=nodes_gdf[node_attr].max())
            sm = ScalarMappable(norm=norm, cmap=node_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, label=node_attr)
    else:
        nodes_gdf.plot(ax=ax, color=node_color, markersize=node_size, alpha=0.8)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=nodes_gdf.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Graph: {graph.name}' if graph.name else 'Graph')
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_graph_by_type(graph, figsize=(12, 10), node_size=50, edge_width=1,
                         add_basemap=False, title=None, ax=None):
    """
    Plot a GeoGraph with nodes and edges colored by their types.
    
    Parameters
    ----------
    graph : GeoGraph
        Graph to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    node_size : int or float, optional
        Size of nodes
    edge_width : int or float, optional
        Width of edges
    add_basemap : bool, optional
        Whether to add a basemap
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    # Convert graph to GeoDataFrames
    nodes_gdf, edges_gdf = graph.to_geodataframes()
    
    # Initialize plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap for node types
    if 'node_type' in nodes_gdf.columns:
        node_types = nodes_gdf['node_type'].unique()
        node_colors = plt.cm.tab10(np.linspace(0, 1, len(node_types)))
        node_color_dict = dict(zip(node_types, node_colors))
        
        # Plot each node type with a different color
        for node_type, color in node_color_dict.items():
            nodes_subset = nodes_gdf[nodes_gdf['node_type'] == node_type]
            nodes_subset.plot(ax=ax, color=color, markersize=node_size, 
                             label=f'Node: {node_type}', alpha=0.8)
    else:
        # If no node_type column, use a default color
        nodes_gdf.plot(ax=ax, color='blue', markersize=node_size, 
                      label='Nodes', alpha=0.8)
    
    # Create a colormap for edge types
    if 'edge_type' in edges_gdf.columns:
        edge_types = edges_gdf['edge_type'].unique()
        edge_colors = plt.cm.tab10(np.linspace(0, 1, len(edge_types)))
        edge_color_dict = dict(zip(edge_types, edge_colors))
        
        # Plot each edge type with a different color
        for edge_type, color in edge_color_dict.items():
            edges_subset = edges_gdf[edges_gdf['edge_type'] == edge_type]
            edges_subset.plot(ax=ax, color=color, linewidth=edge_width, 
                             label=f'Edge: {edge_type}', alpha=0.7)
    else:
        # If no edge_type column, use a default color
        edges_gdf.plot(ax=ax, color='gray', linewidth=edge_width, 
                      label='Edges', alpha=0.7)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=nodes_gdf.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add legend
    ax.legend(loc='best')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Graph by Type: {graph.name}' if graph.name else 'Graph by Type')
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_multilayer_graph(multi_graph, figsize=(16, 12), layer_ids=None, title=None):
    """
    Plot a MultiLayerGeoGraph with each layer in a separate subplot.
    
    Parameters
    ----------
    multi_graph : MultiLayerGeoGraph
        Multi-layer graph to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    layer_ids : list, optional
        List of specific layer IDs to plot (default is all layers)
    title : str, optional
        Overall plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    # Determine which layers to plot
    if layer_ids is None:
        layer_ids = list(multi_graph.layers.keys())
    else:
        # Filter out any layer IDs that don't exist
        layer_ids = [layer_id for layer_id in layer_ids if layer_id in multi_graph.layers]
        
    # Calculate layout
    n_layers = len(layer_ids)
    if n_layers == 0:
        raise ValueError("No valid layers to plot")
        
    # Calculate grid dimensions
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle the case where there's only one subplot
    if n_layers == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Plot each layer
    for i, layer_id in enumerate(layer_ids):
        if i < len(axes):
            graph = multi_graph.layers[layer_id]
            plot_geograph(graph, ax=axes[i], node_size=30, edge_width=0.5,
                         title=f'Layer: {layer_id}')
    
    # Hide any unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Multi-Layer Graph: {multi_graph.name}' if multi_graph.name else 'Multi-Layer Graph', 
                    fontsize=16)
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_graph_connections(graph, node_id, max_degree=2, figsize=(12, 10), 
                             node_size=50, edge_width=1, title=None, ax=None):
    """
    Plot a node and its connections up to a specified degree of separation.
    
    Parameters
    ----------
    graph : GeoGraph
        Graph to plot
    node_id : str or int
        ID of the node to center the plot on
    max_degree : int, optional
        Maximum degree of separation to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    node_size : int or float, optional
        Size of nodes
    edge_width : int or float, optional
        Width of edges
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    # Check if node exists
    if node_id not in graph.nodes:
        raise ValueError(f"Node {node_id} does not exist in the graph")
    
    # Convert graph to GeoDataFrames
    nodes_gdf, edges_gdf = graph.to_geodataframes()
    
    # Initialize sets for tracking nodes and edges
    nodes_to_include = {node_id}
    edges_to_include = set()
    
    # Find connected nodes up to max_degree
    current_nodes = {node_id}
    for degree in range(max_degree):
        new_nodes = set()
        for current_node in current_nodes:
            # Find edges connected to this node
            connected_edges = [
                edge_id for edge_id, edge in graph.edges.items()
                if edge.source == current_node or edge.target == current_node
            ]
            
            for edge_id in connected_edges:
                edge = graph.edges[edge_id]
                edges_to_include.add(edge_id)
                
                # Add connected node
                connected_node = edge.target if edge.source == current_node else edge.source
                if connected_node not in nodes_to_include:
                    new_nodes.add(connected_node)
                    nodes_to_include.add(connected_node)
                    
        # Update current nodes for next iteration
        current_nodes = new_nodes
        if not current_nodes:
            break
    
    # Filter GeoDataFrames to include only the selected nodes and edges
    nodes_subset = nodes_gdf[nodes_gdf['id'].isin(nodes_to_include)]
    edges_subset = edges_gdf[edges_gdf['id'].isin(edges_to_include)]
    
    # Initialize plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create color map for degree of separation
    node_colors = {}
    node_colors[node_id] = 'red'  # Center node
    
    current_nodes = {node_id}
    nodes_at_degree = {0: {node_id}}
    
    for degree in range(1, max_degree + 1):
        nodes_at_degree[degree] = set()
        for current_node in current_nodes:
            # Find edges connected to this node
            connected_edges = [
                edge_id for edge_id, edge in graph.edges.items()
                if edge.source == current_node or edge.target == current_node
            ]
            
            for edge_id in connected_edges:
                edge = graph.edges[edge_id]
                connected_node = edge.target if edge.source == current_node else edge.source
                if connected_node not in node_colors:
                    nodes_at_degree[degree].add(connected_node)
                    # Use a gradient of colors based on degree
                    node_colors[connected_node] = plt.cm.viridis(degree / (max_degree + 1))
        
        current_nodes = nodes_at_degree[degree]
        if not current_nodes:
            break
    
    # Plot edges
    edges_subset.plot(ax=ax, color='gray', linewidth=edge_width, alpha=0.7)
    
    # Plot nodes with colors based on degree of separation
    for node_id, color in node_colors.items():
        node_row = nodes_subset[nodes_subset['id'] == node_id]
        if not node_row.empty:
            node_row.plot(ax=ax, color=color, markersize=node_size, alpha=0.8)
    
    # Add legend for degrees of separation
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                                 markersize=10, label='Center Node')]
    
    for degree in range(1, max_degree + 1):
        if nodes_at_degree.get(degree):
            color = plt.cm.viridis(degree / (max_degree + 1))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color, markersize=10,
                                             label=f'Degree {degree}'))
    
    ax.legend(handles=legend_elements, loc='best')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Connections for Node {node_id} (Max Degree: {max_degree})')
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax 