"""
Functions for loading geospatial data from various formats.
"""

import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from ..core.data_model import TerrainModel, UrbanLayer


def load_geodataframe(filepath, layer=None):
    """
    Load a GeoDataFrame from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the file
    layer : str, optional
        Layer name for multi-layer files (e.g., GeoPackage)
        
    Returns
    -------
    GeoDataFrame
        Loaded data
    """
    # Determine file extension
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() in ['.gpkg', '.sqlite']:
        # GeoPackage or SpatiaLite
        if layer is None:
            # List available layers if none specified
            available_layers = gpd.io.file.fiona.listlayers(filepath)
            if len(available_layers) == 0:
                raise ValueError(f"No layers found in {filepath}")
            elif len(available_layers) == 1:
                layer = available_layers[0]
            else:
                raise ValueError(f"Multiple layers found in {filepath}, please specify one: {available_layers}")
        
        return gpd.read_file(filepath, layer=layer)
    elif ext.lower() in ['.shp', '.geojson', '.json']:
        # Shapefile or GeoJSON
        return gpd.read_file(filepath)
    elif ext.lower() in ['.csv']:
        # CSV (assumes lat/lon columns)
        df = pd.read_csv(filepath)
        
        # Look for common lat/lon column names
        lat_cols = ['lat', 'latitude', 'y']
        lon_cols = ['lon', 'long', 'longitude', 'x']
        
        lat_col = next((col for col in lat_cols if col in df.columns), None)
        lon_col = next((col for col in lon_cols if col in df.columns), None)
        
        if lat_col is None or lon_col is None:
            raise ValueError("Could not identify latitude and longitude columns in CSV")
        
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
        )
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def list_layers(filepath):
    """
    List available layers in a multi-layer geospatial file.
    
    Parameters
    ----------
    filepath : str
        Path to the file
        
    Returns
    -------
    list
        List of layer names
    """
    return gpd.io.file.fiona.listlayers(filepath)


def load_raster(filepath):
    """
    Load a raster dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the raster file
        
    Returns
    -------
    tuple
        (data, transform, crs) where data is a numpy array,
        transform is the affine transformation, and crs is
        the coordinate reference system
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        
    return data, transform, crs


def load_terrain_model(dem_path=None, contours_path=None, contours_layer=None, name=None):
    """
    Load terrain data into a TerrainModel.
    
    Parameters
    ----------
    dem_path : str, optional
        Path to DEM raster file
    contours_path : str, optional
        Path to contour lines vector file
    contours_layer : str, optional
        Layer name for contours in multi-layer files
    name : str, optional
        Name for the terrain model
        
    Returns
    -------
    TerrainModel
        Terrain model with loaded data
    """
    terrain = TerrainModel(name=name)
    
    # Load DEM if provided
    if dem_path:
        dem_data, dem_transform, dem_crs = load_raster(dem_path)
        terrain.set_dem(dem_data, dem_transform)
        terrain.crs = dem_crs
        
        # Calculate extent from transform and shape
        height, width = dem_data.shape
        left, top = dem_transform * (0, 0)
        right, bottom = dem_transform * (width, height)
        terrain.set_extent((left, bottom, right, top))
    
    # Load contours if provided
    if contours_path:
        contours = load_geodataframe(contours_path, layer=contours_layer)
        
        # Check for elevation column
        elev_cols = ['elevation', 'elev', 'height', 'z', 'alt', 'altitude']
        elev_col = next((col for col in elev_cols if col in contours.columns), None)
        
        if elev_col is None and 'contour' in contours.columns:
            # Some datasets store elevation in 'contour' column
            elev_col = 'contour'
            
        if elev_col is None:
            print("Warning: Could not identify elevation column in contours data")
        else:
            # Ensure elevation is numeric
            contours[elev_col] = pd.to_numeric(contours[elev_col], errors='coerce')
        
        terrain.set_contours(contours)
        
        # If no DEM was loaded, use contours CRS
        if not terrain.crs and contours.crs:
            terrain.crs = contours.crs
    
    return terrain


def load_urban_layer(filepath, layer=None, feature_type=None, name=None):
    """
    Load urban features into an UrbanLayer.
    
    Parameters
    ----------
    filepath : str
        Path to the file containing urban features
    layer : str, optional
        Layer name for multi-layer files
    feature_type : str, optional
        Type of features ('building', 'road', etc.)
        If not provided, will be inferred from layer name or geometry type
    name : str, optional
        Name for the layer
        
    Returns
    -------
    UrbanLayer
        Layer containing the loaded features
    """
    # Load data
    gdf = load_geodataframe(filepath, layer=layer)
    
    # Determine feature type if not provided
    if feature_type is None:
        if layer:
            # Try to infer from layer name
            if any(keyword in layer.lower() for keyword in ['build', 'edif']):
                feature_type = 'building'
            elif any(keyword in layer.lower() for keyword in ['road', 'street', 'highway']):
                feature_type = 'road'
            elif any(keyword in layer.lower() for keyword in ['land', 'cover', 'use']):
                feature_type = 'landcover'
            else:
                # Infer from geometry type
                geom_types = gdf.geometry.type.unique()
                if 'Polygon' in geom_types or 'MultiPolygon' in geom_types:
                    if len(gdf) > 100:  # Arbitrary threshold
                        feature_type = 'building'
                    else:
                        feature_type = 'landcover'
                elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
                    feature_type = 'road'
                elif 'Point' in geom_types or 'MultiPoint' in geom_types:
                    feature_type = 'poi'  # Point of interest
                else:
                    feature_type = 'generic'
        else:
            feature_type = 'generic'
    
    # Determine name if not provided
    if name is None:
        if layer:
            name = layer
        else:
            base = os.path.basename(filepath)
            name, _ = os.path.splitext(base)
    
    # Create and return layer
    return UrbanLayer.from_geodataframe(gdf, name, feature_type)


def load_osm_data(bbox, output_path=None, layers=None):
    """
    Load OpenStreetMap data for an area.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (minx, miny, maxx, maxy) in WGS84 coordinates
    output_path : str, optional
        Path to save extracted data
    layers : list, optional
        List of layers to extract (e.g., ['buildings', 'roads'])
        
    Returns
    -------
    dict
        Dictionary of GeoDataFrames keyed by layer name
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("osmnx package is required for OSM data loading. Install it with: pip install osmnx")
    
    # Default layers to extract
    if layers is None:
        layers = ['buildings', 'roads', 'landuse']
    
    # Convert bbox to osmnx format if needed
    if len(bbox) == 4:
        north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
    else:
        north, south, east, west = bbox
    
    # Initialize results
    results = {}
    
    # Extract layers
    if 'buildings' in layers:
        gdf_buildings = ox.features_from_bbox(north, south, east, west, tags={'building': True})
        if not gdf_buildings.empty:
            results['buildings'] = gdf_buildings
    
    if 'roads' in layers:
        gdf_roads = ox.features_from_bbox(north, south, east, west, tags={'highway': True})
        if not gdf_roads.empty:
            results['roads'] = gdf_roads
    
    if 'landuse' in layers:
        gdf_landuse = ox.features_from_bbox(north, south, east, west, tags={'landuse': True})
        if not gdf_landuse.empty:
            results['landuse'] = gdf_landuse
    
    # Save to file if requested
    if output_path:
        for name, gdf in results.items():
            if not gdf.empty:
                if output_path.endswith('.gpkg'):
                    gdf.to_file(output_path, layer=name, driver='GPKG')
                else:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save to individual files
                    base, ext = os.path.splitext(output_path)
                    if not ext:
                        ext = '.geojson'
                    gdf.to_file(f"{base}_{name}{ext}")
    
    return results 