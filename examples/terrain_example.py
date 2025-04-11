"""
Example script for loading and visualizing terrain data.

This script demonstrates how to:
1. Load contour data
2. Create a digital elevation model
3. Visualize the terrain in 2D and 3D
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grapho_terrain as gt
from grapho_terrain.core.data_model import TerrainModel
from grapho_terrain.io import load_geodataframe
from grapho_terrain.processing import create_dem_from_contours
from grapho_terrain.visualization import plot_terrain_model, plot_terrain_3d


def main():
    """Run the terrain example."""
    print("Grapho Terrain - Example Script for Terrain Visualization")
    print("--------------------------------------------------------")
    
    # Create a sample TerrainModel with synthetic contour data
    print("\n1. Creating synthetic contour data...")
    
    # Create a synthetic hills terrain with multiple circular contours
    from shapely.geometry import Point, LineString
    import geopandas as gpd
    
    # Generate concentric circular contours
    contours_data = []
    
    # Center of the terrain
    center_x, center_y = 0, 0
    
    # Create multiple hills
    hills = [
        {'center': (0, 0), 'max_elevation': 1000, 'radius_factor': 1.0},
        {'center': (2000, 1000), 'max_elevation': 800, 'radius_factor': 0.8},
        {'center': (-1500, -1000), 'max_elevation': 600, 'radius_factor': 0.6},
    ]
    
    for hill in hills:
        cx, cy = hill['center']
        max_elev = hill['max_elevation']
        radius_factor = hill['radius_factor']
        
        # Create contours for this hill
        for elevation in range(100, max_elev + 1, 100):
            # Circle radius decreases with elevation
            radius = radius_factor * (max_elev - elevation + 100) * 5
            
            # Create points around the circle
            theta = np.linspace(0, 2*np.pi, 100)
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            
            # Create a LineString and add to contours
            contour = LineString(zip(x, y))
            contours_data.append({
                'geometry': contour,
                'elevation': elevation
            })
    
    # Create GeoDataFrame from the contours
    contours_gdf = gpd.GeoDataFrame(contours_data, crs="EPSG:3857")
    print(f"Created {len(contours_gdf)} synthetic contour lines")
    
    # Create a TerrainModel
    print("\n2. Creating a TerrainModel from contours...")
    terrain = TerrainModel(name="Synthetic Hills")
    terrain.set_contours(contours_gdf)
    
    # Create a DEM from the contours
    print("\n3. Generating a Digital Elevation Model (DEM)...")
    resolution = 25  # 25 meters
    dem_data, dem_transform = create_dem_from_contours(contours_gdf, resolution=resolution)
    terrain.set_dem(dem_data, dem_transform, resolution)
    
    # Calculate extent from contours
    minx, miny, maxx, maxy = contours_gdf.total_bounds
    terrain.set_extent((minx, miny, maxx, maxy))
    
    print(f"DEM dimensions: {dem_data.shape}")
    print(f"DEM resolution: {resolution} meters")
    print(f"Terrain extent: ({minx}, {miny}, {maxx}, {maxy})")
    
    # Visualize the terrain
    print("\n4. Visualizing the terrain...")
    
    # Create 2D visualization
    fig1 = plot_terrain_model(terrain, figsize=(12, 6))
    
    # Create 3D visualization
    fig2 = plot_terrain_3d(terrain, vertical_exaggeration=3)
    
    # Show the plots
    plt.show()
    
    print("\nExample completed successfully!")
    return terrain


if __name__ == "__main__":
    terrain = main() 