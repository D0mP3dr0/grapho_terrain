"""
Graph data structures for geospatial data.

This module provides classes for representing and manipulating
geospatial graphs with multiple layers and heterogeneous node types.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union


class GeoNode:
    """
    A node in a geospatial graph with geographical coordinates.
    """
    
    def __init__(self, node_id, geometry, attributes=None, node_type=None):
        """
        Initialize a GeoNode.
        
        Parameters
        ----------
        node_id : str or int
            Unique identifier for the node
        geometry : shapely.geometry
            Geometry of the node (point, polygon, etc.)
        attributes : dict, optional
            Dictionary of node attributes
        node_type : str, optional
            Type of the node (e.g., 'terrain', 'building', 'road')
        """
        self.id = node_id
        self.geometry = geometry
        self.attributes = attributes or {}
        self.node_type = node_type
        
    def __repr__(self):
        return f"GeoNode(id={self.id}, type={self.node_type})"


class GeoEdge:
    """
    An edge in a geospatial graph connecting two GeoNodes.
    """
    
    def __init__(self, edge_id, source, target, weight=1.0, geometry=None, attributes=None, edge_type=None):
        """
        Initialize a GeoEdge.
        
        Parameters
        ----------
        edge_id : str or int
            Unique identifier for the edge
        source : GeoNode or str or int
            Source node or node ID
        target : GeoNode or str or int
            Target node or node ID
        weight : float, optional
            Weight of the edge
        geometry : shapely.geometry.LineString, optional
            Geometry of the edge
        attributes : dict, optional
            Dictionary of edge attributes
        edge_type : str, optional
            Type of the edge (e.g., 'spatial', 'network', 'functional')
        """
        self.id = edge_id
        self.source = source
        self.target = target
        self.weight = weight
        self.geometry = geometry
        self.attributes = attributes or {}
        self.edge_type = edge_type
        
    def __repr__(self):
        return f"GeoEdge(id={self.id}, source={self.source}, target={self.target}, type={self.edge_type})"


class GeoGraph:
    """
    A geospatial graph with nodes and edges having geographical properties.
    """
    
    def __init__(self, name=None):
        """
        Initialize a GeoGraph.
        
        Parameters
        ----------
        name : str, optional
            Name of the graph
        """
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.node_types = set()
        self.edge_types = set()
        
    def add_node(self, node):
        """
        Add a node to the graph.
        
        Parameters
        ----------
        node : GeoNode
            Node to add
        """
        self.nodes[node.id] = node
        if node.node_type:
            self.node_types.add(node.node_type)
            
    def add_edge(self, edge):
        """
        Add an edge to the graph.
        
        Parameters
        ----------
        edge : GeoEdge
            Edge to add
        """
        self.edges[edge.id] = edge
        if edge.edge_type:
            self.edge_types.add(edge.edge_type)
            
    def get_node(self, node_id):
        """
        Get a node by ID.
        
        Parameters
        ----------
        node_id : str or int
            ID of the node
            
        Returns
        -------
        GeoNode
            The node with the given ID
        """
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id):
        """
        Get an edge by ID.
        
        Parameters
        ----------
        edge_id : str or int
            ID of the edge
            
        Returns
        -------
        GeoEdge
            The edge with the given ID
        """
        return self.edges.get(edge_id)
    
    def to_geodataframes(self):
        """
        Convert the graph to GeoDataFrames.
        
        Returns
        -------
        tuple of GeoDataFrame
            (nodes_gdf, edges_gdf)
        """
        # Create nodes GeoDataFrame
        nodes_data = []
        for node_id, node in self.nodes.items():
            node_data = {"id": node_id, "geometry": node.geometry, "node_type": node.node_type}
            node_data.update(node.attributes)
            nodes_data.append(node_data)
        
        # Create edges GeoDataFrame
        edges_data = []
        for edge_id, edge in self.edges.items():
            edge_data = {
                "id": edge_id,
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "geometry": edge.geometry,
                "edge_type": edge.edge_type
            }
            edge_data.update(edge.attributes)
            edges_data.append(edge_data)
        
        # Create GeoDataFrames
        nodes_gdf = gpd.GeoDataFrame(nodes_data)
        edges_gdf = gpd.GeoDataFrame(edges_data)
        
        return nodes_gdf, edges_gdf
    
    @classmethod
    def from_geodataframes(cls, nodes_gdf, edges_gdf, name=None):
        """
        Create a GeoGraph from GeoDataFrames.
        
        Parameters
        ----------
        nodes_gdf : GeoDataFrame
            GeoDataFrame containing node data
        edges_gdf : GeoDataFrame
            GeoDataFrame containing edge data
        name : str, optional
            Name of the graph
            
        Returns
        -------
        GeoGraph
            A new GeoGraph instance
        """
        graph = cls(name=name)
        
        # Add nodes
        for _, row in nodes_gdf.iterrows():
            node_id = row.get('id')
            geometry = row.geometry
            node_type = row.get('node_type')
            
            # Get attributes (exclude id, geometry, and node_type)
            attributes = row.drop(['id', 'geometry', 'node_type']).to_dict()
            
            node = GeoNode(node_id, geometry, attributes, node_type)
            graph.add_node(node)
        
        # Add edges
        for _, row in edges_gdf.iterrows():
            edge_id = row.get('id')
            source = row.get('source')
            target = row.get('target')
            weight = row.get('weight', 1.0)
            geometry = row.geometry
            edge_type = row.get('edge_type')
            
            # Get attributes (exclude standard fields)
            attributes = row.drop(['id', 'source', 'target', 'weight', 'geometry', 'edge_type']).to_dict()
            
            edge = GeoEdge(edge_id, source, target, weight, geometry, attributes, edge_type)
            graph.add_edge(edge)
            
        return graph


class MultiLayerGeoGraph:
    """
    A multi-layer geospatial graph that contains multiple GeoGraph instances.
    """
    
    def __init__(self, name=None):
        """
        Initialize a MultiLayerGeoGraph.
        
        Parameters
        ----------
        name : str, optional
            Name of the multi-layer graph
        """
        self.name = name
        self.layers = {}
        self.interlayer_edges = {}
        
    def add_layer(self, layer_id, graph):
        """
        Add a layer to the multi-layer graph.
        
        Parameters
        ----------
        layer_id : str or int
            Unique identifier for the layer
        graph : GeoGraph
            The graph to add as a layer
        """
        self.layers[layer_id] = graph
        
    def add_interlayer_edge(self, edge_id, source_layer, source_node, target_layer, target_node, 
                           weight=1.0, attributes=None):
        """
        Add an edge between nodes in different layers.
        
        Parameters
        ----------
        edge_id : str or int
            Unique identifier for the edge
        source_layer : str or int
            ID of the source layer
        source_node : str or int
            ID of the source node
        target_layer : str or int
            ID of the target layer
        target_node : str or int
            ID of the target node
        weight : float, optional
            Weight of the edge
        attributes : dict, optional
            Dictionary of edge attributes
        """
        if source_layer not in self.layers or target_layer not in self.layers:
            raise ValueError("Source or target layer does not exist")
            
        edge = {
            'id': edge_id,
            'source_layer': source_layer,
            'source_node': source_node,
            'target_layer': target_layer,
            'target_node': target_node,
            'weight': weight,
            'attributes': attributes or {}
        }
        
        self.interlayer_edges[edge_id] = edge
        
    def get_layer(self, layer_id):
        """
        Get a layer by ID.
        
        Parameters
        ----------
        layer_id : str or int
            ID of the layer
            
        Returns
        -------
        GeoGraph
            The layer with the given ID
        """
        return self.layers.get(layer_id)
    
    def to_unified_graph(self):
        """
        Convert the multi-layer graph to a unified GeoGraph.
        
        Returns
        -------
        GeoGraph
            A single graph containing all nodes and edges
        """
        unified_graph = GeoGraph(name=f"{self.name}_unified")
        
        # Add nodes from all layers
        for layer_id, graph in self.layers.items():
            for node_id, node in graph.nodes.items():
                # Create a new node with modified ID to avoid conflicts
                unified_id = f"{layer_id}_{node_id}"
                unified_node = GeoNode(
                    unified_id, 
                    node.geometry, 
                    {**node.attributes, 'layer': layer_id}, 
                    node.node_type
                )
                unified_graph.add_node(unified_node)
        
        # Add edges from all layers
        for layer_id, graph in self.layers.items():
            for edge_id, edge in graph.edges.items():
                # Create a new edge with modified IDs
                unified_id = f"{layer_id}_{edge_id}"
                unified_source = f"{layer_id}_{edge.source}"
                unified_target = f"{layer_id}_{edge.target}"
                
                unified_edge = GeoEdge(
                    unified_id,
                    unified_source,
                    unified_target,
                    edge.weight,
                    edge.geometry,
                    {**edge.attributes, 'layer': layer_id},
                    edge.edge_type
                )
                unified_graph.add_edge(unified_edge)
        
        # Add interlayer edges
        for edge_id, edge in self.interlayer_edges.items():
            unified_source = f"{edge['source_layer']}_{edge['source_node']}"
            unified_target = f"{edge['target_layer']}_{edge['target_node']}"
            
            interlayer_edge = GeoEdge(
                edge_id,
                unified_source,
                unified_target,
                edge['weight'],
                None,  # No geometry for interlayer edges
                {**edge['attributes'], 'interlayer': True},
                'interlayer'
            )
            unified_graph.add_edge(interlayer_edge)
            
        return unified_graph 