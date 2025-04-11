"""
Functions for processing terrain data.

This module provides functions for creating, analyzing, and manipulating
terrain data such as digital elevation models and contour lines.
"""

import numpy as np
import geopandas as gpd
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from rasterio.transform import from_origin
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
from ..core.data_model import TerrainModel


def create_dem_from_contours(contours_gdf, resolution=30, method='linear',
                             elev_column='elevation', fill_nodata=True):
    """
    Create a Digital Elevation Model (DEM) from contour lines.
    
    Parameters
    ----------
    contours_gdf : GeoDataFrame
        GeoDataFrame containing contour lines with elevation data
    resolution : float, optional
        Resolution of the output DEM in units of the CRS
    method : str, optional
        Interpolation method ('linear', 'cubic', or 'nearest')
    elev_column : str, optional
        Name of the column containing elevation values
    fill_nodata : bool, optional
        Whether to fill nodata areas using nearest neighbor interpolation
        
    Returns
    -------
    tuple
        (dem_array, transform) where dem_array is a 2D numpy array
        containing the DEM and transform is an affine transformation
    """
    # Validate input
    if not isinstance(contours_gdf, gpd.GeoDataFrame):
        raise ValueError("contours_gdf must be a GeoDataFrame")
        
    if elev_column not in contours_gdf.columns:
        raise ValueError(f"Elevation column '{elev_column}' not found in contours_gdf")
    
    # Get the extent of the contours
    minx, miny, maxx, maxy = contours_gdf.total_bounds
    
    # Add a small buffer to the extent to ensure all contours are included
    buffer = resolution * 2
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    
    # Calculate grid dimensions
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    # Create coordinate grids
    x_grid = np.linspace(minx, maxx, width)
    y_grid = np.linspace(miny, maxy, height)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Extract points and elevations from contours
    points = []
    elevations = []
    
    for _, row in contours_gdf.iterrows():
        geom = row.geometry
        elev = row[elev_column]
        
        # Skip rows with null elevation or geometry
        if geom is None or pd.isna(elev):
            continue
            
        # Ensure numeric elevation
        try:
            elev = float(elev)
        except (ValueError, TypeError):
            continue
            
        # Extract points from the contour geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            for x, y in coords:
                points.append([x, y])
                elevations.append(elev)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                for x, y in coords:
                    points.append([x, y])
                    elevations.append(elev)
    
    # Convert to numpy arrays
    points = np.array(points)
    elevations = np.array(elevations)
    
    if len(points) == 0:
        raise ValueError("No valid points extracted from contours")
    
    # Create the DEM using interpolation
    if method == 'linear':
        # Use LinearNDInterpolator for more control
        interpolator = LinearNDInterpolator(points, elevations)
        Z = interpolator((X, Y))
    elif method == 'cubic':
        # Use griddata for cubic interpolation
        Z = griddata(points, elevations, (X, Y), method='cubic')
    elif method == 'nearest':
        # Use NearestNDInterpolator for more control
        interpolator = NearestNDInterpolator(points, elevations)
        Z = interpolator((X, Y))
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    
    # Fill nodata areas if requested
    if fill_nodata and np.isnan(Z).any():
        # Create a nearest neighbor interpolator to fill nodata areas
        nn_interpolator = NearestNDInterpolator(points, elevations)
        
        # Find indices of NaN values
        nan_indices = np.where(np.isnan(Z))
        
        # Extract coordinates of NaN values
        nan_coords = np.column_stack((X[nan_indices], Y[nan_indices]))
        
        # Interpolate values for NaN coordinates
        nan_values = nn_interpolator(nan_coords)
        
        # Replace NaN values
        Z[nan_indices] = nan_values
    
    # Create the affine transform for the raster
    transform = from_origin(minx, maxy, resolution, resolution)
    
    return Z, transform


def generate_contours(dem_data, transform, interval=50, base=0):
    """
    Generate contour lines from a DEM.
    
    Parameters
    ----------
    dem_data : numpy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
    interval : float, optional
        Contour interval in the same units as the DEM
    base : float, optional
        Base value for contour generation
        
    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing contour lines
    """
    # Generate contour levels
    levels = np.arange(
        base,
        np.nanmax(dem_data) + interval,
        interval
    )
    
    # Create a meshgrid for the DEM
    ny, nx = dem_data.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize lists for contours
    contours_data = []
    
    # Generate contours using matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    contour_set = ax.contour(X, Y, dem_data, levels=levels)
    plt.close(fig)  # Close the figure as we don't need to display it
    
    # Extract contours
    for i, level in enumerate(contour_set.levels):
        paths = contour_set.collections[i].get_paths()
        
        for path in paths:
            vertices = path.vertices
            
            # Convert grid coordinates to real-world coordinates
            real_coords = []
            for j in range(len(vertices)):
                x_grid, y_grid = vertices[j]
                x_world, y_world = transform * (x_grid, y_grid)
                real_coords.append((x_world, y_world))
            
            # Create a LineString from the coordinates
            if len(real_coords) > 1:
                linestring = LineString(real_coords)
                contours_data.append({
                    'geometry': linestring,
                    'elevation': level
                })
    
    # Create a GeoDataFrame
    contours_gdf = gpd.GeoDataFrame(contours_data)
    
    return contours_gdf


def calculate_slope(dem_data, transform, output_units='degrees'):
    """
    Calculate slope from a DEM.
    
    Parameters
    ----------
    dem_data : numpy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
    output_units : str, optional
        Units for the output slope ('degrees' or 'percent')
        
    Returns
    -------
    numpy.ndarray
        2D array containing slope values
    """
    # Get the resolution from the transform
    dx = transform.a
    dy = -transform.e  # Negative because the transform uses a different coordinate system
    
    # Calculate gradients
    gy, gx = np.gradient(dem_data)
    
    # Convert to real-world units
    gx /= dx
    gy /= dy
    
    # Calculate slope
    slope = np.sqrt(gx**2 + gy**2)
    
    # Convert to requested units
    if output_units == 'degrees':
        slope = np.degrees(np.arctan(slope))
    elif output_units == 'percent':
        slope = slope * 100
    else:
        raise ValueError(f"Unsupported output units: {output_units}")
    
    return slope


def calculate_aspect(dem_data, transform):
    """
    Calculate aspect from a DEM.
    
    Parameters
    ----------
    dem_data : numpy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
        
    Returns
    -------
    numpy.ndarray
        2D array containing aspect values in degrees (0-360, clockwise from north)
    """
    # Calculate gradients
    gy, gx = np.gradient(dem_data)
    
    # Calculate aspect (in radians)
    aspect = np.arctan2(-gx, gy)  # Note the negative gx for correct orientation
    
    # Convert to degrees
    aspect = np.degrees(aspect)
    
    # Convert to 0-360 degrees (clockwise from north)
    aspect = 90.0 - aspect
    aspect[aspect < 0] += 360.0
    aspect[aspect > 360] -= 360.0
    
    return aspect


def fill_depressions(dem_data):
    """
    Fill depressions in a DEM to remove pits and ensure proper flow direction.
    
    Parameters
    ----------
    dem_data : numpy.ndarray
        2D array containing elevation data
        
    Returns
    -------
    numpy.ndarray
        2D array with depressions filled
    """
    try:
        import skimage.morphology as morph
        from skimage.morphology import reconstruction
    except ImportError:
        raise ImportError("scikit-image is required for depression filling. Install it with: pip install scikit-image")
    
    # Create a copy of the DEM
    dem_filled = dem_data.copy()
    
    # Replace NaN values with a very small number for processing
    mask = np.isnan(dem_filled)
    dem_filled[mask] = np.nanmin(dem_filled) - 1000
    
    # Create a seed image by subtracting a small value from the maxima
    seed = dem_filled.copy()
    seed[1:-1, 1:-1] = dem_filled.max()
    
    # Perform morphological reconstruction
    dem_filled = reconstruction(seed, dem_filled, method='erosion')
    
    # Restore NaN values
    dem_filled[mask] = np.nan
    
    return dem_filled


def extract_watersheds(dem_data, transform, min_watershed_size=100):
    """
    Extract watersheds from a DEM.
    
    Parameters
    ----------
    dem_data : numpy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
    min_watershed_size : int, optional
        Minimum number of cells for a watershed to be included
        
    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing watershed polygons
    """
    try:
        import skimage.segmentation as seg
        import skimage.morphology as morph
    except ImportError:
        raise ImportError("scikit-image is required for watershed extraction. Install it with: pip install scikit-image")
    
    # Fill depressions in the DEM
    dem_filled = fill_depressions(dem_data)
    
    # Calculate flow direction using gradient
    gy, gx = np.gradient(dem_filled)
    
    # Combine gradients into a single gradient magnitude
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    # Apply watershed segmentation
    watershed_labels = seg.watershed(gradient_mag, markers=None, connectivity=1)
    
    # Convert watershed labels to polygons
    watersheds_data = []
    
    # Process each unique watershed label
    for label in np.unique(watershed_labels):
        if label == 0:  # Skip background
            continue
            
        # Create a binary mask for this watershed
        mask = watershed_labels == label
        
        # Skip small watersheds
        if mask.sum() < min_watershed_size:
            continue
            
        # Find contours of the watershed
        try:
            from skimage import measure
            contours = measure.find_contours(mask.astype(float), 0.5)
        except ImportError:
            raise ImportError("scikit-image is required for contour finding. Install it with: pip install scikit-image")
            
        if len(contours) == 0:
            continue
            
        # Use the largest contour
        contour = max(contours, key=len)
        
        # Convert grid coordinates to real-world coordinates
        real_coords = []
        for y, x in contour:
            x_world, y_world = transform * (x, y)
            real_coords.append((x_world, y_world))
            
        # Create a Polygon from the coordinates
        if len(real_coords) > 3:  # Need at least 4 points for a valid polygon
            try:
                polygon = Polygon(real_coords)
                if polygon.is_valid:
                    # Calculate mean elevation in the watershed
                    mean_elev = np.mean(dem_filled[mask])
                    watersheds_data.append({
                        'geometry': polygon,
                        'watershed_id': int(label),
                        'area': polygon.area,
                        'mean_elevation': float(mean_elev)
                    })
            except Exception as e:
                print(f"Error creating polygon: {e}")
    
    # Create a GeoDataFrame
    watersheds_gdf = gpd.GeoDataFrame(watersheds_data)
    
    return watersheds_gdf 