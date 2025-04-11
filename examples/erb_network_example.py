"""
Example script for cellular network (ERB) analysis.

This script demonstrates how to:
1. Load and process ERB data
2. Create coverage sectors for ERBs
3. Analyze network properties of the cellular network
4. Visualize ERB locations and coverage
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grapho_terrain as gt
from grapho_terrain.telecommunications.erb import ERB, ERBLayer
from grapho_terrain.telecommunications.network import create_erb_graph, analyze_network_metrics
from grapho_terrain.visualization.urban import plot_erb_locations, plot_erb_coverage


def create_sample_erb_data(num_points=30, boundary=(-46.65, -23.55, -46.60, -23.50)):
    """Create sample ERB data for demonstration."""
    # Create random points within boundary
    np.random.seed(42)  # For reproducibility
    
    longitudes = np.random.uniform(boundary[0], boundary[2], num_points)
    latitudes = np.random.uniform(boundary[1], boundary[3], num_points)
    
    # Create a list of points and attributes
    data = []
    
    # Define sample operators and technologies
    operators = ['Claro', 'Vivo', 'TIM', 'Oi']
    technologies = ['4G', '5G']
    frequencies = [700, 850, 1800, 2100, 2600, 3500]
    
    for i in range(num_points):
        # Randomly assign attributes
        operator = np.random.choice(operators)
        technology = np.random.choice(technologies)
        frequency = np.random.choice(frequencies)
        power = np.random.uniform(10, 50)  # Power in watts
        gain = np.random.uniform(12, 18)   # Antenna gain in dBi
        azimuth = np.random.uniform(0, 360)  # Azimuth in degrees
        height = np.random.uniform(15, 45)   # Height in meters
        
        data.append({
            'id': f'ERB_{i+1}',
            'geometry': Point(longitudes[i], latitudes[i]),
            'operadora': operator,
            'tecnologia': technology,
            'freq_mhz': frequency,
            'potencia_watts': power,
            'ganho_antena': gain,
            'azimute': azimuth,
            'altura_m': height
        })
    
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert to GeoDataFrame
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    return gdf


def main():
    """Run the ERB network analysis example."""
    print("Grapho Terrain - Example Script for ERB Network Analysis")
    print("------------------------------------------------------")
    
    # Create sample ERB data
    print("\n1. Creating sample ERB data...")
    gdf_erb = create_sample_erb_data(num_points=30)
    print(f"Created {len(gdf_erb)} sample ERB points")
    
    # Create an ERB layer
    print("\n2. Creating ERB layer...")
    erb_layer = ERBLayer.from_geodataframe(gdf_erb, name="Sample ERBs")
    print(f"Created ERB layer with {len(erb_layer.erbs)} ERBs")
    
    # Create coverage sectors
    print("\n3. Creating coverage sectors...")
    gdf_sectors = erb_layer.create_coverage_sectors(tipo_area='urbana')
    print(f"Created {len(gdf_sectors)} coverage sectors")
    
    # Create ERB graph
    print("\n4. Creating ERB network graph...")
    erb_graph = create_erb_graph(erb_layer, k_nearest=3)
    print(f"Created graph with {len(erb_graph.nodes)} nodes and {len(erb_graph.edges)} edges")
    
    # Analyze network metrics
    print("\n5. Analyzing network metrics...")
    metrics = analyze_network_metrics(erb_graph)
    print("Network metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Visualize the results
    print("\n6. Creating visualizations...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot ERB locations by operator
    plot_erb_locations(gdf_erb, ax=axes[0], by_operator=True, add_basemap=True,
                     title="ERB Locations by Operator")
    
    # Plot ERB coverage areas
    plot_erb_coverage(gdf_erb, gdf_sectors, ax=axes[1], by_operator=True, add_basemap=True,
                    title="Coverage Areas by Operator")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "erb_network_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nOutput saved to: {output_path}")
    print("\nShowing plots (close window to continue)...")
    plt.show()
    
    print("\nExample completed successfully!")
    return erb_layer, gdf_sectors, erb_graph


if __name__ == "__main__":
    erb_layer, gdf_sectors, erb_graph = main() 