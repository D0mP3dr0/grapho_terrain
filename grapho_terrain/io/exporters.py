"""
Functions for exporting geospatial data to various formats.
"""

import os
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import numpy as np
from ..core.data_model import TerrainModel, UrbanLayer
from ..core.graph import GeoGraph, MultiLayerGeoGraph


def export_geodataframe(gdf, filepath, layer=None, driver=None):
    """
    Export a GeoDataFrame to a file.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Data to export
    filepath : str
        Path to the output file
    layer : str, optional
        Layer name for multi-layer formats
    driver : str, optional
        Driver name for fiona output (auto-detected from extension if not provided)
    """
    # Determine file extension
    _, ext = os.path.splitext(filepath)
    
    # Determine driver if not provided
    if driver is None:
        if ext.lower() == '.gpkg':
            driver = 'GPKG'
        elif ext.lower() == '.sqlite':
            driver = 'SQLite'
        elif ext.lower() == '.shp':
            driver = 'ESRI Shapefile'
        elif ext.lower() in ['.geojson', '.json']:
            driver = 'GeoJSON'
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Export the data
    if layer and driver in ['GPKG', 'SQLite']:
        gdf.to_file(filepath, layer=layer, driver=driver)
    else:
        gdf.to_file(filepath, driver=driver)


def export_raster(data, transform, crs, filepath, nodata=None):
    """
    Export a raster dataset to a file.
    
    Parameters
    ----------
    data : numpy.ndarray
        Raster data as a 2D array
    transform : affine.Affine
        Affine transform for the raster
    crs : str or dict
        Coordinate reference system
    filepath : str
        Path to the output file
    nodata : float, optional
        Value to use for nodata pixels
    """
    # Determine file extension
    _, ext = os.path.splitext(filepath)
    
    # Set default nodata value if not provided
    if nodata is None:
        if data.dtype == np.float32 or data.dtype == np.float64:
            nodata = np.nan
        else:
            nodata = 0
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Write the raster
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data, 1)


def export_terrain_model(terrain, output_dir, base_name=None):
    """
    Export a TerrainModel to files.
    
    Parameters
    ----------
    terrain : TerrainModel
        Terrain model to export
    output_dir : str
        Directory to save the output files
    base_name : str, optional
        Base name for output files (default is terrain.name or 'terrain')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base name
    if base_name is None:
        base_name = terrain.name if terrain.name else 'terrain'
    
    # Export DEM if available
    if terrain.dem is not None and terrain.dem_transform is not None:
        dem_path = os.path.join(output_dir, f"{base_name}_dem.tif")
        export_raster(terrain.dem, terrain.dem_transform, terrain.crs, dem_path)
    
    # Export contours if available
    if terrain.contours is not None:
        contours_path = os.path.join(output_dir, f"{base_name}_contours.gpkg")
        export_geodataframe(terrain.contours, contours_path, layer='contours')
    
    # Export metadata if available
    if terrain.metadata:
        import json
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(terrain.metadata, f, indent=2)


def export_urban_layer(layer, filepath, layer_name=None):
    """
    Export an UrbanLayer to a file.
    
    Parameters
    ----------
    layer : UrbanLayer
        Urban layer to export
    filepath : str
        Path to the output file
    layer_name : str, optional
        Layer name for multi-layer formats (default is layer.name)
    """
    # Convert to GeoDataFrame
    gdf = layer.to_geodataframe()
    
    # Determine layer name
    if layer_name is None:
        layer_name = layer.name
    
    # Export the data
    export_geodataframe(gdf, filepath, layer=layer_name)


def export_graph(graph, output_dir, base_name=None):
    """
    Export a GeoGraph to files.
    
    Parameters
    ----------
    graph : GeoGraph
        Graph to export
    output_dir : str
        Directory to save the output files
    base_name : str, optional
        Base name for output files (default is graph.name or 'graph')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base name
    if base_name is None:
        base_name = graph.name if graph.name else 'graph'
    
    # Convert to GeoDataFrames
    nodes_gdf, edges_gdf = graph.to_geodataframes()
    
    # Export nodes
    nodes_path = os.path.join(output_dir, f"{base_name}_nodes.gpkg")
    export_geodataframe(nodes_gdf, nodes_path, layer='nodes')
    
    # Export edges
    edges_path = os.path.join(output_dir, f"{base_name}_edges.gpkg")
    export_geodataframe(edges_gdf, edges_path, layer='edges')


def export_multi_layer_graph(graph, output_dir, base_name=None):
    """
    Export a MultiLayerGeoGraph to files.
    
    Parameters
    ----------
    graph : MultiLayerGeoGraph
        Multi-layer graph to export
    output_dir : str
        Directory to save the output files
    base_name : str, optional
        Base name for output files (default is graph.name or 'multi_graph')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base name
    if base_name is None:
        base_name = graph.name if graph.name else 'multi_graph'
    
    # Export each layer
    for layer_id, layer_graph in graph.layers.items():
        layer_dir = os.path.join(output_dir, f"layer_{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)
        export_graph(layer_graph, layer_dir, base_name=f"{base_name}_layer_{layer_id}")
    
    # Export interlayer edges if any
    if graph.interlayer_edges:
        import pandas as pd
        
        # Convert interlayer edges to DataFrame
        edges_data = []
        for edge_id, edge in graph.interlayer_edges.items():
            data = {
                "id": edge_id,
                "source_layer": edge['source_layer'],
                "source_node": edge['source_node'],
                "target_layer": edge['target_layer'],
                "target_node": edge['target_node'],
                "weight": edge['weight']
            }
            data.update(edge['attributes'])
            edges_data.append(data)
        
        # Create DataFrame and save to CSV
        edges_df = pd.DataFrame(edges_data)
        edges_path = os.path.join(output_dir, f"{base_name}_interlayer_edges.csv")
        edges_df.to_csv(edges_path, index=False)
    
    # Export unified graph
    unified_graph = graph.to_unified_graph()
    unified_dir = os.path.join(output_dir, "unified")
    os.makedirs(unified_dir, exist_ok=True)
    export_graph(unified_graph, unified_dir, base_name=f"{base_name}_unified") 