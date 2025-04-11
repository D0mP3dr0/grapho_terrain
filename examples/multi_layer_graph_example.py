"""
Example script for multi-layer feature-rich geospatial graph analysis.

This script demonstrates how to:
1. Create multi-layer graphs from different geospatial data sources
2. Add feature matrices to graph nodes and edges
3. Create connections between different layers
4. Export to PyTorch Geometric format for machine learning
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grapho_terrain as gt
from grapho_terrain.network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
from grapho_terrain.core.graph import GeoNode, GeoEdge
from grapho_terrain.core.data_model import TerrainModel, UrbanLayer
from grapho_terrain.telecommunications.erb import ERBLayer
from grapho_terrain.visualization.network import plot_geograph

def create_sample_layers(num_buildings=30, num_roads=10, num_erbs=15, 
                        boundary=(-46.65, -23.55, -46.60, -23.50)):
    """Create sample geospatial layers for demonstration."""
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a sample buildings layer
    buildings_data = []
    for i in range(num_buildings):
        # Random point within boundary
        lon = np.random.uniform(boundary[0], boundary[2])
        lat = np.random.uniform(boundary[1], boundary[3])
        
        # Create a small square polygon around the point
        size = np.random.uniform(0.0005, 0.001)
        polygon = Polygon([
            (lon - size, lat - size),
            (lon + size, lat - size),
            (lon + size, lat + size),
            (lon - size, lat + size),
            (lon - size, lat - size)
        ])
        
        # Random building attributes
        height = np.random.uniform(5, 50)
        floors = np.random.randint(1, 15)
        building_types = ['residential', 'commercial', 'industrial', 'public']
        building_type = np.random.choice(building_types)
        
        buildings_data.append({
            'id': f'BLDG_{i+1}',
            'geometry': polygon,
            'height': height,
            'floors': floors,
            'building_type': building_type,
            'year_built': np.random.randint(1950, 2023)
        })
    
    # Create buildings GeoDataFrame
    gdf_buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")
    
    # Create a sample roads layer
    roads_data = []
    for i in range(num_roads):
        # Random start and end points within boundary
        start_lon = np.random.uniform(boundary[0], boundary[2])
        start_lat = np.random.uniform(boundary[1], boundary[3])
        
        # Random angle and length
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0.005, 0.015)
        
        # Calculate end point
        end_lon = start_lon + length * np.cos(angle)
        end_lat = start_lat + length * np.sin(angle)
        
        # Ensure end point is within boundary
        end_lon = np.clip(end_lon, boundary[0], boundary[2])
        end_lat = np.clip(end_lat, boundary[1], boundary[3])
        
        # Create LineString
        line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
        
        # Random road attributes
        road_types = ['primary', 'secondary', 'residential', 'service']
        road_type = np.random.choice(road_types)
        
        lanes = np.random.randint(1, 5)
        
        roads_data.append({
            'id': f'ROAD_{i+1}',
            'geometry': line,
            'road_type': road_type,
            'lanes': lanes,
            'speed_limit': np.random.choice([30, 40, 60, 80]),
            'length_km': line.length * 111.32  # Approximate conversion to km
        })
    
    # Create roads GeoDataFrame
    gdf_roads = gpd.GeoDataFrame(roads_data, crs="EPSG:4326")
    
    # Create a sample ERBs layer
    erbs_data = []
    for i in range(num_erbs):
        # Random point within boundary
        lon = np.random.uniform(boundary[0], boundary[2])
        lat = np.random.uniform(boundary[1], boundary[3])
        
        # Create a point geometry
        point = Point(lon, lat)
        
        # Random ERB attributes
        operators = ['Claro', 'Vivo', 'TIM', 'Oi']
        operator = np.random.choice(operators)
        
        technologies = ['4G', '5G']
        technology = np.random.choice(technologies)
        
        frequencies = [700, 850, 1800, 2100, 2600, 3500]
        frequency = np.random.choice(frequencies)
        
        erbs_data.append({
            'id': f'ERB_{i+1}',
            'geometry': point,
            'operadora': operator,
            'tecnologia': technology,
            'freq_mhz': frequency,
            'potencia_watts': np.random.uniform(10, 50),
            'ganho_antena': np.random.uniform(12, 18),
            'azimute': np.random.uniform(0, 360),
            'altura_m': np.random.uniform(15, 45)
        })
    
    # Create ERBs GeoDataFrame
    gdf_erbs = gpd.GeoDataFrame(erbs_data, crs="EPSG:4326")
    
    return gdf_buildings, gdf_roads, gdf_erbs


def main():
    """Run the multi-layer graph example."""
    print("Grapho Terrain - Example Script for Multi-Layer Feature Graph Analysis")
    print("-------------------------------------------------------------------")
    
    # Create sample data layers
    print("\n1. Creating sample data layers...")
    gdf_buildings, gdf_roads, gdf_erbs = create_sample_layers()
    print(f"Created {len(gdf_buildings)} buildings, {len(gdf_roads)} roads, and {len(gdf_erbs)} ERBs")
    
    # Create a multi-layer feature graph
    print("\n2. Creating a multi-layer feature graph...")
    
    # Create a dictionary of dataframes
    dataframes = {
        'buildings': gdf_buildings,
        'roads': gdf_roads,
        'erbs': gdf_erbs
    }
    
    # Specify features for each layer
    features = {
        'buildings': ['height', 'floors', 'year_built'],
        'roads': ['lanes', 'speed_limit', 'length_km'],
        'erbs': ['freq_mhz', 'potencia_watts', 'ganho_antena', 'altura_m']
    }
    
    # Create the multi-layer graph
    multi_graph = MultiLayerFeatureGraph.from_layer_dataframes(
        dataframes, features=features, k_intra=3, k_inter=2
    )
    
    print(f"Created multi-layer graph with {len(multi_graph.layers)} layers")
    for layer_id, layer in multi_graph.layers.items():
        print(f"  Layer '{layer_id}': {len(layer.nodes)} nodes, {len(layer.edges)} edges")
        
    # Count interlayer edges
    interlayer_edges_count = {}
    for edge_id, edge in multi_graph.interlayer_edges.items():
        layer_pair = (edge['source_layer'], edge['target_layer'])
        interlayer_edges_count[layer_pair] = interlayer_edges_count.get(layer_pair, 0) + 1
        
    print("Interlayer connections:")
    for layer_pair, count in interlayer_edges_count.items():
        print(f"  {layer_pair[0]} -> {layer_pair[1]}: {count} edges")
    
    # Examine feature matrices
    print("\n3. Examining feature matrices...")
    for layer_id, layer in multi_graph.layers.items():
        if layer.node_features is not None:
            print(f"  Layer '{layer_id}' node features shape: {layer.node_features.shape}")
            print(f"  Feature names: {layer.node_feature_names}")
    
    # Convert to PyTorch Geometric format
    print("\n4. Converting to PyTorch Geometric format...")
    hetero_data = multi_graph.to_torch_geometric_hetero()
    print("Created HeteroData object with:")
    print(f"  Node types: {hetero_data.node_types}")
    print(f"  Edge types: {hetero_data.edge_types}")
    for node_type in hetero_data.node_types:
        print(f"  '{node_type}' nodes feature shape: {hetero_data[node_type].x.shape}")
    
    # Visualize the layers
    print("\n5. Visualizing the layers...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each layer
    for i, (layer_id, layer) in enumerate(multi_graph.layers.items()):
        if i < len(axes):
            plot_geograph(layer, ax=axes[i], node_attr=layer.node_feature_names[0] if layer.node_features is not None else None,
                       title=f"Layer: {layer_id}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multi_layer_graph.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nOutput saved to: {output_path}")
    print("\nShowing plots (close window to continue)...")
    plt.show()
    
    print("\nExample completed successfully!")
    return multi_graph, hetero_data


if __name__ == "__main__":
    multi_graph, hetero_data = main() 