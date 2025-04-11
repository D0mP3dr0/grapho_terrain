"""
Network analysis for telecommunications systems.

This module provides functions for modeling and analyzing 
telecommunications networks as graphs, particularly cellular networks.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import networkx as nx
from ..core.graph import GeoGraph, GeoNode, GeoEdge

def create_erb_graph(erb_layer, max_distance=None, k_nearest=None):
    """
    Create a graph from ERB data.
    
    Parameters
    ----------
    erb_layer : ERBLayer
        Layer containing ERB objects
    max_distance : float, optional
        Maximum distance for connecting ERBs (in km)
    k_nearest : int, optional
        Number of nearest neighbors to connect each ERB to
        
    Returns
    -------
    GeoGraph
        Graph representing the ERB network
    """
    # Convert ERB layer to a GeoDataFrame
    gdf = erb_layer.to_geodataframe()
    
    # Create a graph
    graph = GeoGraph(name="ERB Network")
    
    # Add nodes for each ERB
    for erb_id, erb in erb_layer.erbs.items():
        # Create a node with ERB attributes
        node = GeoNode(erb_id, erb.geometry, erb.attributes, "erb")
        graph.add_node(node)
    
    # Create edges based on specified criteria
    if max_distance is not None:
        # Create edges based on maximum distance
        for erb1_id, erb1 in erb_layer.erbs.items():
            for erb2_id, erb2 in erb_layer.erbs.items():
                if erb1_id != erb2_id:
                    # Calculate distance between ERBs (approximate)
                    lon1, lat1 = erb1.geometry.x, erb1.geometry.y
                    lon2, lat2 = erb2.geometry.x, erb2.geometry.y
                    
                    # Simple approximation using Euclidean distance
                    # More accurate distance calculation would be preferred for real applications
                    dx = (lon2 - lon1) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
                    dy = (lat2 - lat1) * 111.32
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    
                    if distance <= max_distance:
                        # Create a LineString for the edge geometry
                        line = LineString([(lon1, lat1), (lon2, lat2)])
                        
                        # Create an edge
                        edge_id = f"{erb1_id}_{erb2_id}"
                        edge = GeoEdge(
                            edge_id=edge_id,
                            source=erb1_id,
                            target=erb2_id,
                            weight=distance,
                            geometry=line,
                            attributes={"distance_km": distance},
                            edge_type="distance"
                        )
                        
                        graph.add_edge(edge)
                        
    elif k_nearest is not None:
        # Create edges based on k nearest neighbors
        for erb1_id, erb1 in erb_layer.erbs.items():
            # Calculate distances to all other ERBs
            distances = []
            for erb2_id, erb2 in erb_layer.erbs.items():
                if erb1_id != erb2_id:
                    # Calculate distance between ERBs (approximate)
                    lon1, lat1 = erb1.geometry.x, erb1.geometry.y
                    lon2, lat2 = erb2.geometry.x, erb2.geometry.y
                    
                    # Simple approximation using Euclidean distance
                    dx = (lon2 - lon1) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
                    dy = (lat2 - lat1) * 111.32
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    
                    distances.append((erb2_id, distance))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Connect to k nearest neighbors
            for i in range(min(k_nearest, len(distances))):
                erb2_id, distance = distances[i]
                
                # Get coordinates
                lon1, lat1 = erb1.geometry.x, erb1.geometry.y
                erb2 = erb_layer.erbs[erb2_id]
                lon2, lat2 = erb2.geometry.x, erb2.geometry.y
                
                # Create a LineString for the edge geometry
                line = LineString([(lon1, lat1), (lon2, lat2)])
                
                # Create an edge
                edge_id = f"{erb1_id}_{erb2_id}"
                edge = GeoEdge(
                    edge_id=edge_id,
                    source=erb1_id,
                    target=erb2_id,
                    weight=distance,
                    geometry=line,
                    attributes={"distance_km": distance},
                    edge_type="nearest"
                )
                
                graph.add_edge(edge)
    
    return graph


def create_voronoi_coverage(erb_layer, boundary_polygon=None):
    """
    Create Voronoi polygons representing theoretical coverage areas.
    
    Parameters
    ----------
    erb_layer : ERBLayer
        Layer containing ERB objects
    boundary_polygon : shapely.geometry.Polygon, optional
        Boundary polygon to clip the Voronoi diagram
        
    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing Voronoi polygons
    """
    # Convert ERB layer to a GeoDataFrame
    gdf = erb_layer.to_geodataframe()
    
    # Extract coordinates for Voronoi computation
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
    
    # If less than 4 points, Voronoi computation will fail
    if len(coords) < 4:
        raise ValueError("At least 4 points are required for Voronoi computation")
    
    # Compute Voronoi diagram
    vor = Voronoi(coords)
    
    # Create Voronoi polygons
    from shapely.geometry import Polygon
    
    # Function to create a Voronoi polygon for a given region
    def voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'
            
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions
        vertices : array-like
            Coordinates for revised Voronoi vertices
        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")
            
        new_regions = []
        new_vertices = vor.vertices.tolist()
        
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2
            
        # Construct a map of ridge points to vertices
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
            
        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            
            if all(v >= 0 for v in vertices):
                # Finite region
                new_regions.append(vertices)
                continue
                
            # Reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # Finite ridge: already in the region
                    continue
                    
                # Infinite ridge
                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices.max() + radius * direction
                
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
                
            # Sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            
            new_regions.append(new_region.tolist())
            
        return new_regions, np.asarray(new_vertices)
    
    # Get Voronoi polygons
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # Create GeoDataFrame with Voronoi polygons
    voronoi_polygons = []
    for i, region in enumerate(regions):
        polygon = Polygon(vertices[region])
        
        # Clip to boundary if provided
        if boundary_polygon is not None:
            polygon = polygon.intersection(boundary_polygon)
            
        # Add to list with associated ERB ID
        voronoi_polygons.append({
            'geometry': polygon,
            'erb_id': gdf.iloc[i].name
        })
        
    # Create GeoDataFrame
    voronoi_gdf = gpd.GeoDataFrame(voronoi_polygons, crs=gdf.crs)
    
    # Join with original ERB data
    voronoi_gdf = voronoi_gdf.merge(gdf, left_on='erb_id', right_index=True, suffixes=('', '_erb'))
    
    return voronoi_gdf


def create_coverage_overlap_graph(erb_layer, tipo_area='urbana'):
    """
    Create a graph where ERBs are connected if their coverage areas overlap.
    
    Parameters
    ----------
    erb_layer : ERBLayer
        Layer containing ERB objects
    tipo_area : str, optional
        Area type for coverage calculation ('urbana', 'suburbana', 'rural')
        
    Returns
    -------
    GeoGraph
        Graph representing the ERB network with overlap connections
    """
    # Create coverage sectors
    sectors_gdf = erb_layer.create_coverage_sectors(tipo_area=tipo_area)
    
    # Create a graph
    graph = GeoGraph(name="ERB Coverage Overlap Network")
    
    # Add nodes for each ERB
    for erb_id, erb in erb_layer.erbs.items():
        # Create a node with ERB attributes
        node = GeoNode(erb_id, erb.geometry, erb.attributes, "erb")
        graph.add_node(node)
    
    # Find overlapping coverage areas
    for i, row_i in sectors_gdf.iterrows():
        erb_id_i = row_i['erb_id']
        geom_i = row_i.geometry
        
        for j, row_j in sectors_gdf.iterrows():
            if i < j:  # Avoid duplicate edges and self-connections
                erb_id_j = row_j['erb_id']
                geom_j = row_j.geometry
                
                # Check if coverage areas overlap
                if geom_i.intersects(geom_j):
                    # Calculate overlap area
                    overlap = geom_i.intersection(geom_j)
                    overlap_area = overlap.area
                    
                    # Create a geometry for the edge (a line connecting the ERBs)
                    erb_i = erb_layer.erbs[erb_id_i]
                    erb_j = erb_layer.erbs[erb_id_j]
                    line = LineString([(erb_i.geometry.x, erb_i.geometry.y), 
                                      (erb_j.geometry.x, erb_j.geometry.y)])
                    
                    # Create an edge
                    edge_id = f"{erb_id_i}_{erb_id_j}"
                    edge = GeoEdge(
                        edge_id=edge_id,
                        source=erb_id_i,
                        target=erb_id_j,
                        weight=overlap_area,
                        geometry=line,
                        attributes={"overlap_area": overlap_area},
                        edge_type="overlap"
                    )
                    
                    graph.add_edge(edge)
    
    return graph


def create_networkx_graph(geo_graph):
    """
    Convert a GeoGraph to a NetworkX graph for analysis.
    
    Parameters
    ----------
    geo_graph : GeoGraph
        GeoGraph to convert
        
    Returns
    -------
    networkx.Graph
        NetworkX graph representation
    """
    import networkx as nx
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node_id, node in geo_graph.nodes.items():
        G.add_node(node_id, **node.attributes)
    
    # Add edges
    for edge_id, edge in geo_graph.edges.items():
        G.add_edge(edge.source, edge.target, weight=edge.weight, **edge.attributes)
    
    return G


def analyze_network_metrics(geo_graph):
    """
    Calculate network metrics for a telecommunications graph.
    
    Parameters
    ----------
    geo_graph : GeoGraph
        Graph to analyze
        
    Returns
    -------
    dict
        Dictionary of network metrics
    """
    # Convert to NetworkX graph
    G = create_networkx_graph(geo_graph)
    
    # Calculate basic metrics
    metrics = {}
    
    # Number of nodes and edges
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    
    # Connectivity
    if nx.is_connected(G):
        metrics['is_connected'] = True
        metrics['diameter'] = nx.diameter(G)
        metrics['radius'] = nx.radius(G)
    else:
        metrics['is_connected'] = False
        components = list(nx.connected_components(G))
        metrics['num_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len))
    
    # Node degree statistics
    degrees = [d for _, d in G.degree()]
    metrics['avg_degree'] = sum(degrees) / len(degrees)
    metrics['min_degree'] = min(degrees)
    metrics['max_degree'] = max(degrees)
    
    # Centrality measures
    # Only calculate for smaller networks as these can be computationally intensive
    if metrics['num_nodes'] <= 1000:
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        metrics['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality)
        metrics['max_degree_centrality_node'] = max(degree_centrality, key=degree_centrality.get)
        
        # Betweenness centrality (only for smaller networks)
        if metrics['num_nodes'] <= 500:
            betweenness_centrality = nx.betweenness_centrality(G)
            metrics['avg_betweenness_centrality'] = sum(betweenness_centrality.values()) / len(betweenness_centrality)
            metrics['max_betweenness_centrality_node'] = max(betweenness_centrality, key=betweenness_centrality.get)
    
    # Clustering coefficient
    metrics['avg_clustering'] = nx.average_clustering(G)
    
    # Edge weight statistics if applicable
    if nx.get_edge_attributes(G, 'weight'):
        weights = [w for _, _, w in G.edges(data='weight')]
        metrics['avg_edge_weight'] = sum(weights) / len(weights)
        metrics['min_edge_weight'] = min(weights)
        metrics['max_edge_weight'] = max(weights)
    
    return metrics 