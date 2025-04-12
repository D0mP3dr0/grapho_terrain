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
import warnings

# Optional RAPIDS/GPU imports
try:
    import cupy
    import cucim.skimage as cu_morph
    import cucim.skimage.segmentation as cu_seg
    import cucim.skimage.measure as cu_measure
    # cupyx.scipy.ndimage might be needed for gradient if numpy version differs significantly
    # from cupyx.scipy.ndimage import gradient as cupy_gradient # Alternative gradient
    RAPIDS_AVAILABLE = True
    warnings.warn("RAPIDS/GPU libraries (CuPy, cucim.skimage) found. GPU acceleration will be available for terrain processing.")
except ImportError:
    RAPIDS_AVAILABLE = False
    warnings.warn("RAPIDS/GPU libraries (CuPy, cucim.skimage) not found. GPU acceleration will NOT be available for terrain processing.")
    cupy = None # Define as None if not available
    cu_morph = None
    cu_seg = None
    cu_measure = None

# CPU-only imports (ensure they are still available)
try:
    import skimage.morphology as morph
    import skimage.segmentation as seg
    import skimage.measure as measure
    from skimage.morphology import reconstruction
    SCIKIT_IMAGE_AVAILABLE = True
except ImportError:
    SCIKIT_IMAGE_AVAILABLE = False
    morph = None
    seg = None
    measure = None
    reconstruction = None
    warnings.warn("Scikit-image not found. Some CPU-based terrain functions (fill_depressions, extract_watersheds) will not be available.")


def create_dem_from_contours(contours_gdf, resolution=30, method='linear',
                             elev_column='elevation', fill_nodata=True):
    """
    Create a Digital Elevation Model (DEM) from contour lines.
    Uses SciPy interpolation (CPU only).

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
    ------
    tuple
        (dem_array, transform) where dem_array is a 2D numpy array
        containing the DEM and transform is an affine transformation
    """
    warnings.warn("create_dem_from_contours currently runs on CPU using SciPy.")
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

    # Create the DEM using interpolation (SciPy on CPU)
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

    # Fill nodata areas if requested (using SciPy on CPU)
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
    Uses Matplotlib (CPU only).

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
    ------
    GeoDataFrame
        GeoDataFrame containing contour lines
    """
    warnings.warn("generate_contours runs on CPU using Matplotlib.")
    # Ensure dem_data is numpy array on CPU
    if cupy and isinstance(dem_data, cupy.ndarray):
        dem_data = dem_data.get()

    # Generate contour levels
    min_elev = np.nanmin(dem_data)
    max_elev = np.nanmax(dem_data)
    levels = np.arange(
        base + np.floor((min_elev - base) / interval) * interval, # Start level aligned with interval below min
        max_elev + interval,
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
    try:
        # Handle potential NaN values in dem_data for contouring
        masked_dem = np.ma.masked_invalid(dem_data)
        contour_set = ax.contour(X, Y, masked_dem, levels=levels)
    finally:
        plt.close(fig)  # Ensure figure is closed even if contour fails

    # Extract contours
    for i, level in enumerate(contour_set.levels):
        # Check if the collection has paths (it might not for certain levels)
        if i < len(contour_set.collections):
            collection = contour_set.collections[i]
            paths = collection.get_paths()

            for path in paths:
                vertices = path.vertices

                # Convert grid coordinates to real-world coordinates
                real_coords = []
                for j in range(len(vertices)):
                    x_grid, y_grid = vertices[j]
                    # Ensure indices are within bounds (important for edge contours)
                    x_grid = np.clip(x_grid, 0, nx - 1)
                    y_grid = np.clip(y_grid, 0, ny - 1)
                    # Apply affine transform
                    x_world, y_world = transform * (x_grid, y_grid)
                    real_coords.append((x_world, y_world))

                # Create a LineString from the coordinates
                if len(real_coords) > 1:
                    try:
                        linestring = LineString(real_coords)
                        if linestring.is_valid and not linestring.is_empty:
                             contours_data.append({
                                 'geometry': linestring,
                                 'elevation': level
                             })
                    except Exception as e:
                         warnings.warn(f"Could not create valid LineString for contour level {level}: {e}")


    # Create a GeoDataFrame
    if contours_data:
        contours_gdf = gpd.GeoDataFrame(contours_data, geometry='geometry')
        # Optionally set CRS if known (e.g., from input or DEM)
        # contours_gdf.crs = ...
    else:
        # Return empty GeoDataFrame if no contours were generated
        contours_gdf = gpd.GeoDataFrame({'geometry': [], 'elevation': []}, geometry='geometry')


    return contours_gdf


def calculate_slope(dem_data, transform, output_units='degrees', use_gpu=False):
    """
    Calculate slope from a DEM. Uses CuPy for gradient calculation if use_gpu=True.

    Parameters
    ----------
    dem_data : numpy.ndarray or cupy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
    output_units : str, optional
        Units for the output slope ('degrees' or 'percent')
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False)

    Returns
    ------
    numpy.ndarray
        2D array containing slope values (always returned as numpy array on CPU)
    """
    # Get the resolution from the transform
    dx = transform.a
    dy = -transform.e  # Negative because the transform uses a different coordinate system

    if dx == 0 or dy == 0:
        raise ValueError("Invalid transform: dx or dy is zero.")

    # Select execution path
    if use_gpu and RAPIDS_AVAILABLE:
        warnings.warn("Calculating slope on GPU using CuPy.")
        xp = cupy
        try:
            # Ensure data is on GPU
            if not isinstance(dem_data, xp.ndarray):
                dem_gpu = xp.asarray(dem_data)
            else:
                dem_gpu = dem_data

            # Calculate gradients on GPU
            gy, gx = xp.gradient(dem_gpu)

            # Convert to real-world units
            gx /= dx
            gy /= dy

            # Calculate slope
            slope_gpu = xp.sqrt(gx**2 + gy**2)

            # Convert to requested units
            if output_units == 'degrees':
                slope_gpu = xp.degrees(xp.arctan(slope_gpu))
            elif output_units == 'percent':
                slope_gpu = slope_gpu * 100
            else:
                raise ValueError(f"Unsupported output units: {output_units}")

            # Return result as numpy array on CPU
            return slope_gpu.get()

        except Exception as e:
            warnings.warn(f"GPU slope calculation failed: {e}. Falling back to CPU.")
            # Fallback to CPU if GPU fails
            if cupy and isinstance(dem_data, cupy.ndarray): # Ensure data is numpy for CPU path
                 dem_data = dem_data.get()
            xp = np

    else:
        warnings.warn("Calculating slope on CPU using NumPy.")
        if cupy and isinstance(dem_data, cupy.ndarray): # Ensure data is numpy for CPU path
            dem_data = dem_data.get()
        xp = np


    # CPU Calculation (NumPy)
    gy, gx = xp.gradient(dem_data)

    # Convert to real-world units
    gx /= dx
    gy /= dy

    # Calculate slope
    slope = xp.sqrt(gx**2 + gy**2)

    # Convert to requested units
    if output_units == 'degrees':
        slope = xp.degrees(xp.arctan(slope))
    elif output_units == 'percent':
        slope = slope * 100
    else:
        raise ValueError(f"Unsupported output units: {output_units}")

    return slope


def calculate_aspect(dem_data, transform, use_gpu=False):
    """
    Calculate aspect from a DEM. Uses CuPy for gradient/arctan2 if use_gpu=True.

    Parameters
    ----------
    dem_data : numpy.ndarray or cupy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False)

    Returns
    ------
    numpy.ndarray
        2D array containing aspect values in degrees (0-360, clockwise from north)
        (always returned as numpy array on CPU)
    """
     # Select execution path
    if use_gpu and RAPIDS_AVAILABLE:
        warnings.warn("Calculating aspect on GPU using CuPy.")
        xp = cupy
        try:
            # Ensure data is on GPU
            if not isinstance(dem_data, xp.ndarray):
                dem_gpu = xp.asarray(dem_data)
            else:
                dem_gpu = dem_data

            # Calculate gradients on GPU
            gy, gx = xp.gradient(dem_gpu)

            # Calculate aspect (in radians)
            aspect_gpu = xp.arctan2(-gx, gy)  # Note the negative gx for correct orientation

            # Convert to degrees
            aspect_gpu = xp.degrees(aspect_gpu)

            # Convert to 0-360 degrees (clockwise from north)
            aspect_gpu = 90.0 - aspect_gpu
            aspect_gpu[aspect_gpu < 0] += 360.0
            # aspect_gpu[aspect_gpu > 360] -= 360.0 # This might not be needed if input is within atan2 range

            # Return result as numpy array on CPU
            return aspect_gpu.get()

        except Exception as e:
            warnings.warn(f"GPU aspect calculation failed: {e}. Falling back to CPU.")
            # Fallback to CPU if GPU fails
            if cupy and isinstance(dem_data, cupy.ndarray): # Ensure data is numpy for CPU path
                 dem_data = dem_data.get()
            xp = np
    else:
        warnings.warn("Calculating aspect on CPU using NumPy.")
        if cupy and isinstance(dem_data, cupy.ndarray): # Ensure data is numpy for CPU path
            dem_data = dem_data.get()
        xp = np

    # CPU Calculation (NumPy)
    gy, gx = xp.gradient(dem_data)

    # Calculate aspect (in radians)
    aspect = xp.arctan2(-gx, gy)  # Note the negative gx for correct orientation

    # Convert to degrees
    aspect = xp.degrees(aspect)

    # Convert to 0-360 degrees (clockwise from north)
    aspect = 90.0 - aspect
    aspect[aspect < 0] += 360.0
    # aspect[aspect > 360] -= 360.0 # This might not be needed if input is within atan2 range

    return aspect


def fill_depressions(dem_data, use_gpu=False):
    """
    Fill depressions in a DEM using morphological reconstruction.
    Uses cucim.skimage if use_gpu=True.

    Parameters
    ----------
    dem_data : numpy.ndarray or cupy.ndarray
        2D array containing elevation data
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False)

    Returns
    ------
    numpy.ndarray
        2D array with depressions filled (always returned as numpy array on CPU)
    """
    # Select execution path
    if use_gpu and RAPIDS_AVAILABLE and cu_morph:
        warnings.warn("Filling depressions on GPU using cucim.skimage.")
        xp = cupy
        morph_lib = cu_morph
        try:
            # Ensure data is on GPU
            if not isinstance(dem_data, xp.ndarray):
                dem_gpu = xp.asarray(dem_data, dtype=xp.float32) # Use float32 for efficiency
            else:
                dem_gpu = dem_data.astype(xp.float32)

            # Create a copy
            dem_filled_gpu = dem_gpu.copy()

            # Replace NaN values with a very small number for processing
            # Note: CuPy NaN handling might differ slightly, check if needed
            mask = xp.isnan(dem_filled_gpu)
            min_val = xp.nanmin(dem_filled_gpu)
            if xp.isnan(min_val): min_val = 0 # Handle case where all are NaN
            dem_filled_gpu[mask] = min_val - 1000 # Or xp.finfo(xp.float32).min?

            # Create a seed image by subtracting a small value from the maxima
            # Ensure seed is same shape and type
            seed = dem_filled_gpu.copy()
            max_val = xp.nanmax(dem_filled_gpu)
            if xp.isnan(max_val): max_val = 0 # Handle case where all are NaN
            # Set border pixels to max_val, inner pixels to max_val
            # A common approach is to start from the borders inward
            seed[1:-1, 1:-1] = max_val # Original skimage approach starts seed high internally
            # Alternative: seed = dem_filled_gpu.copy(); seed[0,:]=max_val; seed[-1,:]=max_val; etc.

            # Perform morphological reconstruction (erosion method)
            # Check cucim docs for exact function signature if needed
            dem_filled_gpu = morph_lib.reconstruction(seed, dem_filled_gpu, method='erosion')

            # Restore NaN values
            dem_filled_gpu[mask] = xp.nan

            # Return result as numpy array on CPU
            return dem_filled_gpu.get()

        except Exception as e:
            warnings.warn(f"GPU depression filling failed: {e}. Falling back to CPU.")
            # Fallback to CPU if GPU fails
            if cupy and isinstance(dem_data, cupy.ndarray): # Ensure data is numpy for CPU path
                 dem_data = dem_data.get()
            # Ensure scikit-image is available for CPU fallback
            if not SCIKIT_IMAGE_AVAILABLE:
                 raise ImportError("scikit-image is required for CPU depression filling fallback.")


    # CPU Path
    warnings.warn("Filling depressions on CPU using scikit-image.")
    if not SCIKIT_IMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for CPU depression filling.")

    if cupy and isinstance(dem_data, cupy.ndarray): # Ensure data is numpy for CPU path
        dem_data = dem_data.get()
    xp = np
    morph_lib = morph

    # Create a copy of the DEM
    dem_filled = dem_data.copy()

    # Replace NaN values with a very small number for processing
    mask = np.isnan(dem_filled)
    min_val = np.nanmin(dem_filled)
    if np.isnan(min_val): min_val = 0
    dem_filled[mask] = min_val - 1000

    # Create a seed image
    seed = dem_filled.copy()
    max_val = np.nanmax(dem_filled)
    if np.isnan(max_val): max_val = 0
    seed[1:-1, 1:-1] = max_val

    # Perform morphological reconstruction
    dem_filled = reconstruction(seed, dem_filled, method='erosion')

    # Restore NaN values
    dem_filled[mask] = np.nan

    return dem_filled


def extract_watersheds(dem_data, transform, min_watershed_size=100, use_gpu=False):
    """
    Extract watersheds from a DEM using watershed segmentation.
    Uses cucim.skimage if use_gpu=True for segmentation and gradient.

    Parameters
    ----------
    dem_data : numpy.ndarray or cupy.ndarray
        2D array containing elevation data
    transform : affine.Affine
        Affine transform for the DEM
    min_watershed_size : int, optional
        Minimum number of cells for a watershed to be included
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False)

    Returns
    ------
    GeoDataFrame
        GeoDataFrame containing watershed polygons
    """
    # Fill depressions first (respecting use_gpu flag)
    try:
        dem_filled = fill_depressions(dem_data, use_gpu=use_gpu)
    except Exception as e:
        warnings.warn(f"Depression filling failed: {e}. Proceeding with original DEM for watershed extraction.")
        # Decide whether to proceed with original or raise error
        if use_gpu and cupy and isinstance(dem_data, cupy.ndarray):
            dem_filled = dem_data # Use original GPU array
        elif not use_gpu and isinstance(dem_data, np.ndarray):
            dem_filled = dem_data # Use original numpy array
        else: # Mixed types or failure, try converting to numpy
            try:
                 dem_filled = dem_data.get() if cupy and isinstance(dem_data, cupy.ndarray) else np.asarray(dem_data)
            except:
                 raise ValueError("Could not prepare DEM data for watershed after depression fill failure.") from e


    # Select execution path for watershed
    if use_gpu and RAPIDS_AVAILABLE and cu_seg and cu_measure:
        warnings.warn("Extracting watersheds on GPU using cucim.skimage.")
        xp = cupy
        seg_lib = cu_seg
        measure_lib = cu_measure
        try:
            # Ensure data is on GPU
            if not isinstance(dem_filled, xp.ndarray):
                dem_filled_gpu = xp.asarray(dem_filled, dtype=xp.float32)
            else:
                dem_filled_gpu = dem_filled.astype(xp.float32)

            # Handle potential NaNs before gradient calculation
            # (Replace with low value or use masked gradient if available)
            nan_mask_gpu = xp.isnan(dem_filled_gpu)
            min_val_gpu = xp.nanmin(dem_filled_gpu)
            if xp.isnan(min_val_gpu): min_val_gpu = 0
            dem_filled_gpu[nan_mask_gpu] = min_val_gpu - 1000

            # Calculate flow direction using gradient on GPU
            gy, gx = xp.gradient(dem_filled_gpu)

            # Combine gradients into a single gradient magnitude
            gradient_mag_gpu = xp.sqrt(gx**2 + gy**2)

            # Apply watershed segmentation on GPU
            # Check cucim docs for marker handling - 'None' might not be supported directly
            # May need to generate markers using local minima or use default
            # markers = None # Placeholder
            # watershed_labels_gpu = seg_lib.watershed(gradient_mag_gpu, markers=markers, connectivity=1)
            watershed_labels_gpu = seg_lib.watershed(gradient_mag_gpu, connectivity=1) # Use default markers

            # Restore NaNs in the original data if needed, watershed labels should be int
            # dem_filled_gpu[nan_mask_gpu] = xp.nan # Restore NaNs if dem_filled_gpu is reused

            # Move labels to CPU for polygonization
            watershed_labels = watershed_labels_gpu.get()

        except Exception as e:
            warnings.warn(f"GPU watershed calculation failed: {e}. Falling back to CPU.")
            # Fallback to CPU if GPU fails
            dem_filled = dem_filled.get() if cupy and isinstance(dem_filled, cupy.ndarray) else np.asarray(dem_filled)
             # Ensure scikit-image is available for CPU fallback
            if not SCIKIT_IMAGE_AVAILABLE:
                 raise ImportError("scikit-image is required for CPU watershed fallback.")
            xp = np
            seg_lib = seg
            measure_lib = measure # For find_contours
            # Proceed with CPU path below

    else: # CPU Path selected or fallback
        warnings.warn("Extracting watersheds on CPU using scikit-image.")
        if not SCIKIT_IMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for watershed extraction.")
        # Ensure data is numpy
        dem_filled = dem_filled.get() if cupy and isinstance(dem_filled, cupy.ndarray) else np.asarray(dem_filled)
        xp = np
        seg_lib = seg
        measure_lib = measure

        # Handle NaNs
        nan_mask = np.isnan(dem_filled)
        min_val = np.nanmin(dem_filled)
        if np.isnan(min_val): min_val = 0
        dem_filled[nan_mask] = min_val - 1000

        # Calculate gradient magnitude on CPU
        gy, gx = np.gradient(dem_filled)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Apply watershed segmentation on CPU
        watershed_labels = seg_lib.watershed(gradient_mag, connectivity=1) # Default markers

        # Restore NaNs in the original data if needed
        # dem_filled[nan_mask] = np.nan # Restore NaNs if dem_filled is reused


    # --- Polygonization (Common path, uses CPU libraries) ---
    watersheds_data = []
    unique_labels = np.unique(watershed_labels)

    # Ensure dem_filled is available on CPU for mean elevation calculation
    dem_filled_cpu = dem_filled.get() if cupy and isinstance(dem_filled, cupy.ndarray) else np.asarray(dem_filled)


    # Process each unique watershed label
    for label in unique_labels:
        if label == 0:  # Skip background/boundary label
            continue

        # Create a binary mask for this watershed
        mask = watershed_labels == label

        # Skip small watersheds
        if mask.sum() < min_watershed_size:
            continue

        # Find contours of the watershed mask using scikit-image (CPU)
        # Ensure mask is float for find_contours
        try:
            # Use scikit-image measure for contours, regardless of GPU/CPU watershed run
            if measure_lib is None: raise ImportError("scikit-image measure unavailable for contours")
            contours = measure_lib.find_contours(mask.astype(float), 0.5)
        except Exception as e:
            warnings.warn(f"Could not find contours for watershed label {label}: {e}")
            continue


        if not contours:
            continue

        # Use the largest contour (most vertices) as the boundary
        contour = max(contours, key=len)

        # Convert grid coordinates to real-world coordinates using the transform
        real_coords = []
        ny, nx = mask.shape
        for y_grid, x_grid in contour:
            # Clip coordinates to be within valid image bounds before transforming
            x_grid_clipped = np.clip(x_grid, 0, nx - 1)
            y_grid_clipped = np.clip(y_grid, 0, ny - 1)
            x_world, y_world = transform * (x_grid_clipped, y_grid_clipped)
            real_coords.append((x_world, y_world))

        # Create a Polygon from the coordinates
        if len(real_coords) > 3:  # Need at least 4 points for a valid polygon
            try:
                polygon = Polygon(real_coords)
                # Optional: Simplify or buffer polygon slightly if needed
                # polygon = polygon.simplify(tolerance=...)
                if polygon.is_valid and not polygon.is_empty:
                    # Calculate mean elevation in the watershed using CPU data
                    mean_elev = np.mean(dem_filled_cpu[mask]) # Use dem_filled_cpu here

                    watersheds_data.append({
                        'geometry': polygon,
                        'watershed_id': int(label),
                        'area_m2': polygon.area, # Area is in CRS units squared
                        'mean_elevation': float(mean_elev) if not np.isnan(mean_elev) else None
                    })
            except Exception as e:
                warnings.warn(f"Error creating polygon for watershed label {label}: {e}")


    # Create a GeoDataFrame
    if watersheds_data:
        watersheds_gdf = gpd.GeoDataFrame(watersheds_data, geometry='geometry')
        # Optionally set CRS
        # watersheds_gdf.crs = ...
    else:
        watersheds_gdf = gpd.GeoDataFrame({
            'geometry': [], 'watershed_id': [], 'area_m2': [], 'mean_elevation': []
        }, geometry='geometry')


    return watersheds_gdf 