"""
Feature-rich graph models for geospatial networks.

This module provides enhanced graph data structures and functions
for handling feature-rich geospatial networks with support for
heterogeneous node types, feature matrices, and multiple layers.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import geopandas as gpd
from ..core.graph import GeoGraph, GeoNode, GeoEdge

class FeatureGeoGraph(GeoGraph):
    """
    A geospatial graph with support for node and edge feature matrices.
    
    This extends the base GeoGraph with additional support for feature
    matrices and compatibility with PyTorch Geometric.
    """
    
    def __init__(self, name=None):
        """
        Initialize a FeatureGeoGraph.
        
        Parameters
        ----------
        name : str, optional
            Name of the graph
        """
        super().__init__(name=name)
        
        # Feature matrices
        self.node_features = None
        self.edge_features = None
        self.node_feature_names = []
        self.edge_feature_names = []
        self.target = None
        
    def set_node_features(self, feature_matrix, feature_names=None):
        """
        Set the node feature matrix.
        
        Parameters
        ----------
        feature_matrix : numpy.ndarray or torch.Tensor
            Feature matrix with shape (num_nodes, num_features)
        feature_names : list, optional
            Names of the features (columns in the matrix)
        """
        # Convert to numpy array if it's not already
        if isinstance(feature_matrix, torch.Tensor):
            feature_matrix = feature_matrix.numpy()
            
        self.node_features = feature_matrix
        
        if feature_names:
            self.node_feature_names = feature_names
        else:
            # Generate default feature names
            self.node_feature_names = [f"feature_{i}" for i in range(feature_matrix.shape[1])]
            
    def set_edge_features(self, feature_matrix, feature_names=None):
        """
        Set the edge feature matrix.
        
        Parameters
        ----------
        feature_matrix : numpy.ndarray or torch.Tensor
            Feature matrix with shape (num_edges, num_features)
        feature_names : list, optional
            Names of the features (columns in the matrix)
        """
        # Convert to numpy array if it's not already
        if isinstance(feature_matrix, torch.Tensor):
            feature_matrix = feature_matrix.numpy()
            
        self.edge_features = feature_matrix
        
        if feature_names:
            self.edge_feature_names = feature_names
        else:
            # Generate default feature names
            self.edge_feature_names = [f"feature_{i}" for i in range(feature_matrix.shape[1])]
            
    def set_target(self, target):
        """
        Set the target variable for supervised learning.
        
        Parameters
        ----------
        target : numpy.ndarray or torch.Tensor
            Target variable, typically with shape (num_nodes,) for node-level tasks
            or a single value for graph-level tasks
        """
        # Convert to numpy array if it's not already
        if isinstance(target, torch.Tensor):
            target = target.numpy()
            
        self.target = target
        
    def create_node_feature_matrix(self, attrs=None, numeric_only=True, fill_value=0):
        """
        Create a node feature matrix from node attributes.
        
        Parameters
        ----------
        attrs : list, optional
            List of attribute names to include in the feature matrix
            If None, all numeric attributes will be used
        numeric_only : bool, optional
            Whether to include only numeric attributes
        fill_value : float, optional
            Value to use for missing attributes
            
        Returns
        -------
        numpy.ndarray
            Node feature matrix
        list
            Names of the features
        """
        # Collect all attributes if not specified
        all_attrs = set()
        
        for node_id, node in self.nodes.items():
            all_attrs.update(node.attributes.keys())
            
        # Filter attributes if needed
        if attrs:
            # Use only specified attributes
            feature_names = [attr for attr in attrs if attr in all_attrs]
        else:
            # Use all attributes (possibly filtering non-numeric ones later)
            feature_names = list(all_attrs)
            
        # Initialize feature matrix
        n_nodes = len(self.nodes)
        n_features = len(feature_names)
        feature_matrix = np.full((n_nodes, n_features), fill_value, dtype=np.float32)
        
        # Fill the feature matrix
        for i, (node_id, node) in enumerate(self.nodes.items()):
            for j, attr in enumerate(feature_names):
                if attr in node.attributes:
                    value = node.attributes[attr]
                    
                    # Handle different types
                    if isinstance(value, (int, float)):
                        feature_matrix[i, j] = value
                    elif numeric_only:
                        # Skip non-numeric values if numeric_only is True
                        pass
                    else:
                        # Try to convert to numeric
                        try:
                            feature_matrix[i, j] = float(value)
                        except (ValueError, TypeError):
                            pass
                            
        # Set the feature matrix and names
        self.set_node_features(feature_matrix, feature_names)
        
        return feature_matrix, feature_names
        
    def create_edge_feature_matrix(self, attrs=None, numeric_only=True, fill_value=0):
        """
        Create an edge feature matrix from edge attributes.
        
        Parameters
        ----------
        attrs : list, optional
            List of attribute names to include in the feature matrix
            If None, all numeric attributes will be used
        numeric_only : bool, optional
            Whether to include only numeric attributes
        fill_value : float, optional
            Value to use for missing attributes
            
        Returns
        -------
        numpy.ndarray
            Edge feature matrix
        list
            Names of the features
        """
        # Collect all attributes if not specified
        all_attrs = set()
        
        for edge_id, edge in self.edges.items():
            all_attrs.update(edge.attributes.keys())
            
        # Filter attributes if needed
        if attrs:
            # Use only specified attributes
            feature_names = [attr for attr in attrs if attr in all_attrs]
        else:
            # Use all attributes (possibly filtering non-numeric ones later)
            feature_names = list(all_attrs)
            
        # Initialize feature matrix
        n_edges = len(self.edges)
        n_features = len(feature_names)
        feature_matrix = np.full((n_edges, n_features), fill_value, dtype=np.float32)
        
        # Fill the feature matrix
        for i, (edge_id, edge) in enumerate(self.edges.items()):
            for j, attr in enumerate(feature_names):
                if attr in edge.attributes:
                    value = edge.attributes[attr]
                    
                    # Handle different types
                    if isinstance(value, (int, float)):
                        feature_matrix[i, j] = value
                    elif numeric_only:
                        # Skip non-numeric values if numeric_only is True
                        pass
                    else:
                        # Try to convert to numeric
                        try:
                            feature_matrix[i, j] = float(value)
                        except (ValueError, TypeError):
                            pass
                            
        # Set the feature matrix and names
        self.set_edge_features(feature_matrix, feature_names)
        
        return feature_matrix, feature_names
        
    def to_torch_geometric(self):
        """
        Convert the graph to a PyTorch Geometric Data object.
        
        Returns
        -------
        torch_geometric.data.Data
            PyTorch Geometric Data object
        """
        # Create edge index
        edge_index = []
        
        # Map node IDs to indices
        node_id_to_index = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        
        # Add edges to edge index
        for edge_id, edge in self.edges.items():
            source_idx = node_id_to_index[edge.source]
            target_idx = node_id_to_index[edge.target]
            edge_index.append([source_idx, target_idx])
            
        # Create edge index tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create node features tensor if available
        if self.node_features is not None:
            x = torch.tensor(self.node_features, dtype=torch.float)
        else:
            # Create dummy features if none are available
            x = torch.ones((len(self.nodes), 1), dtype=torch.float)
            
        # Create edge features tensor if available
        if self.edge_features is not None:
            edge_attr = torch.tensor(self.edge_features, dtype=torch.float)
        else:
            edge_attr = None
            
        # Create target tensor if available
        if self.target is not None:
            y = torch.tensor(self.target, dtype=torch.float)
        else:
            y = None
            
        # Create the PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # Add additional metadata
        data.node_id_to_index = node_id_to_index
        data.node_feature_names = self.node_feature_names
        data.edge_feature_names = self.edge_feature_names
        
        return data


class MultiLayerFeatureGraph:
    """
    A multi-layer graph with support for feature matrices.
    
    This class represents a heterogeneous graph with multiple layers,
    each potentially having different node and edge types with
    different feature spaces.
    """
    
    def __init__(self, name=None):
        """
        Initialize a MultiLayerFeatureGraph.
        
        Parameters
        ----------
        name : str, optional
            Name of the multi-layer graph
        """
        self.name = name
        self.layers = {}  # Dictionary of layer_id -> FeatureGeoGraph
        self.interlayer_edges = {}  # Dictionary of edge_id -> edge_dict
        
    def add_layer(self, layer_id, graph):
        """
        Add a layer to the multi-layer graph.
        
        Parameters
        ----------
        layer_id : str or int
            Unique identifier for the layer
        graph : FeatureGeoGraph or GeoGraph
            The graph to add as a layer (will be converted to FeatureGeoGraph if needed)
        """
        # Convert to FeatureGeoGraph if needed
        if not isinstance(graph, FeatureGeoGraph):
            feature_graph = FeatureGeoGraph(name=graph.name)
            
            # Copy nodes and edges
            for node_id, node in graph.nodes.items():
                feature_graph.add_node(node)
                
            for edge_id, edge in graph.edges.items():
                feature_graph.add_edge(edge)
                
            graph = feature_graph
            
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
            
        # Check if source and target nodes exist in their respective layers
        if source_node not in self.layers[source_layer].nodes:
            raise ValueError(f"Source node {source_node} does not exist in layer {source_layer}")
            
        if target_node not in self.layers[target_layer].nodes:
            raise ValueError(f"Target node {target_node} does not exist in layer {target_layer}")
            
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
        
    def to_torch_geometric_hetero(self):
        """
        Convert the multi-layer graph to a PyTorch Geometric HeteroData object.
        
        Returns
        -------
        torch_geometric.data.HeteroData
            PyTorch Geometric HeteroData object
        """
        # Create a HeteroData object
        data = HeteroData()
        
        # Map from layer ID and node ID to index in the layer
        node_maps = {}
        
        # Add nodes and their features for each layer
        for layer_id, layer in self.layers.items():
            # Create mapping from node ID to index
            node_id_to_index = {node_id: i for i, node_id in enumerate(layer.nodes.keys())}
            node_maps[layer_id] = node_id_to_index
            
            # Get node features or create dummy features
            if layer.node_features is not None:
                x = torch.tensor(layer.node_features, dtype=torch.float)
            else:
                x = torch.ones((len(layer.nodes), 1), dtype=torch.float)
                
            # Add node features to the HeteroData object
            data[str(layer_id)].x = x
            
            # Store node mapping in the data object
            data[str(layer_id)].node_map = node_id_to_index
            
        # Add intra-layer edges for each layer
        for layer_id, layer in self.layers.items():
            # Get the node mapping for this layer
            node_map = node_maps[layer_id]
            
            # Create edge indices for this layer
            edge_indices = []
            edge_attrs = []
            
            for edge_id, edge in layer.edges.items():
                source_idx = node_map[edge.source]
                target_idx = node_map[edge.target]
                edge_indices.append([source_idx, target_idx])
                
                # Add edge attributes if available
                if layer.edge_features is not None:
                    edge_attr = [edge.attributes.get(attr, 0) for attr in layer.edge_feature_names]
                    edge_attrs.append(edge_attr)
                    
            # Add edge indices to the HeteroData object
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                data[str(layer_id), 'to', str(layer_id)].edge_index = edge_index
                
                # Add edge attributes if available
                if edge_attrs:
                    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                    data[str(layer_id), 'to', str(layer_id)].edge_attr = edge_attr
                    
        # Add inter-layer edges
        for edge_id, edge in self.interlayer_edges.items():
            source_layer = str(edge['source_layer'])
            target_layer = str(edge['target_layer'])
            
            # Get source and target node indices
            source_node = edge['source_node']
            target_node = edge['target_node']
            
            source_idx = node_maps[edge['source_layer']][source_node]
            target_idx = node_maps[edge['target_layer']][target_node]
            
            # Create edge type string
            edge_type = (source_layer, 'to', target_layer)
            
            # Initialize edge index for this edge type if not already done
            if edge_type not in data.edge_types:
                data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
                data[edge_type].edge_attr = torch.zeros((0, 1), dtype=torch.float)
                
            # Add edge to edge index
            edge_index = data[edge_type].edge_index
            new_edge = torch.tensor([[source_idx], [target_idx]], dtype=torch.long)
            data[edge_type].edge_index = torch.cat([edge_index, new_edge], dim=1)
            
            # Add edge weight as attribute
            edge_attr = data[edge_type].edge_attr
            new_attr = torch.tensor([[edge['weight']]], dtype=torch.float)
            data[edge_type].edge_attr = torch.cat([edge_attr, new_attr], dim=0)
            
        return data
    
    @classmethod
    def from_layer_dataframes(cls, dataframes, layer_names=None, features=None, k_intra=3, k_inter=2):
        """
        Create a MultiLayerFeatureGraph from a collection of DataFrames.
        
        Parameters
        ----------
        dataframes : dict or list
            Dictionary of layer_id -> GeoDataFrame, or list of GeoDataFrames
        layer_names : list, optional
            Names of layers (used only if dataframes is a list)
        features : dict, optional
            Dictionary of layer_id -> list of feature column names
        k_intra : int, optional
            Number of nearest neighbors for intra-layer connections
        k_inter : int, optional
            Number of nearest neighbors for inter-layer connections
            
        Returns
        -------
        MultiLayerFeatureGraph
            Multi-layer graph constructed from the dataframes
        """
        # Create a multi-layer graph
        multi_graph = cls()
        
        # Convert list to dictionary if needed
        if isinstance(dataframes, list):
            if layer_names is None:
                layer_names = [f"layer_{i}" for i in range(len(dataframes))]
                
            dataframes = dict(zip(layer_names, dataframes))
            
        # Create a layer for each dataframe
        for layer_id, gdf in dataframes.items():
            # Create a graph for this layer
            layer_graph = FeatureGeoGraph(name=str(layer_id))
            
            # Add nodes from the dataframe
            for idx, row in gdf.iterrows():
                # Use the index as node ID if no specific ID column
                node_id = idx
                
                # Extract geometry and attributes
                geom = row.geometry
                
                # Convert row to dictionary of attributes (excluding geometry)
                attributes = row.drop('geometry').to_dict()
                
                # Create a node
                node = GeoNode(node_id, geom, attributes, node_type=str(layer_id))
                layer_graph.add_node(node)
                
            # Create edges within this layer (k nearest neighbors)
            from scipy.spatial.distance import cdist
            
            # Extract node coordinates
            coords = np.array([[node.geometry.x, node.geometry.y] for node in layer_graph.nodes.values()])
            
            # Compute pairwise distances
            distances = cdist(coords, coords)
            
            # For each node, connect to k nearest neighbors
            node_ids = list(layer_graph.nodes.keys())
            
            for i, node_id in enumerate(node_ids):
                # Get indices of k+1 nearest neighbors (including self)
                nearest_indices = np.argsort(distances[i])[:k_intra + 1]
                
                # Connect to each neighbor (excluding self)
                for j in nearest_indices:
                    if i != j:  # Avoid self-loops
                        neighbor_id = node_ids[j]
                        
                        # Create an edge
                        edge_id = f"{node_id}_{neighbor_id}"
                        
                        # Get node coordinates
                        source_node = layer_graph.nodes[node_id]
                        target_node = layer_graph.nodes[neighbor_id]
                        
                        # Create a LineString for the edge geometry
                        from shapely.geometry import LineString
                        line = LineString([(source_node.geometry.x, source_node.geometry.y),
                                          (target_node.geometry.x, target_node.geometry.y)])
                        
                        # Create the edge
                        edge = GeoEdge(
                            edge_id=edge_id,
                            source=node_id,
                            target=neighbor_id,
                            weight=distances[i, j],
                            geometry=line,
                            attributes={"distance": distances[i, j]},
                            edge_type="nearest"
                        )
                        
                        layer_graph.add_edge(edge)
            
            # Create feature matrix if features are specified
            if features is not None and layer_id in features:
                feature_cols = features[layer_id]
                
                # Extract feature values for each node
                feature_values = []
                
                for node_id in layer_graph.nodes.keys():
                    # Get the row in the original dataframe
                    row = gdf.loc[node_id]
                    
                    # Extract feature values
                    node_features = [row[col] if col in row else 0 for col in feature_cols]
                    feature_values.append(node_features)
                    
                # Create feature matrix
                feature_matrix = np.array(feature_values, dtype=np.float32)
                layer_graph.set_node_features(feature_matrix, feature_cols)
            else:
                # Create feature matrix from all numeric attributes
                layer_graph.create_node_feature_matrix(numeric_only=True)
                
            # Add the layer to the multi-layer graph
            multi_graph.add_layer(layer_id, layer_graph)
            
        # Create inter-layer edges
        layer_ids = list(multi_graph.layers.keys())
        
        for i, layer_id1 in enumerate(layer_ids):
            for j, layer_id2 in enumerate(layer_ids):
                if i < j:  # Avoid duplicate connections
                    # Get layer graphs
                    layer1 = multi_graph.layers[layer_id1]
                    layer2 = multi_graph.layers[layer_id2]
                    
                    # Extract node coordinates
                    coords1 = np.array([[node.geometry.x, node.geometry.y] for node in layer1.nodes.values()])
                    coords2 = np.array([[node.geometry.x, node.geometry.y] for node in layer2.nodes.values()])
                    
                    # Compute pairwise distances between layers
                    cross_distances = cdist(coords1, coords2)
                    
                    # For each node in layer1, connect to k nearest nodes in layer2
                    node_ids1 = list(layer1.nodes.keys())
                    node_ids2 = list(layer2.nodes.keys())
                    
                    for ii, node_id1 in enumerate(node_ids1):
                        # Get indices of k nearest nodes in layer2
                        nearest_indices = np.argsort(cross_distances[ii])[:k_inter]
                        
                        # Connect to each nearest node in layer2
                        for jj in nearest_indices:
                            node_id2 = node_ids2[jj]
                            
                            # Create an inter-layer edge
                            edge_id = f"{layer_id1}_{node_id1}_to_{layer_id2}_{node_id2}"
                            
                            # Add the edge
                            multi_graph.add_interlayer_edge(
                                edge_id=edge_id,
                                source_layer=layer_id1,
                                source_node=node_id1,
                                target_layer=layer_id2,
                                target_node=node_id2,
                                weight=cross_distances[ii, jj],
                                attributes={"distance": cross_distances[ii, jj]}
                            )
                            
        return multi_graph 