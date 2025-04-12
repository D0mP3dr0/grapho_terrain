"""
Radio coverage modeling and analysis for telecommunications networks.

This module provides functions for calculating radio coverage from cellular 
base stations (ERBs) considering terrain, frequency, and other factors.
"""

import numpy as np
import math
from shapely.geometry import Point, Polygon
import geopandas as gpd
from ..core.data_model import TerrainModel
import warnings

# Optional RAPIDS/GPU imports
try:
    import cupy as cp
    # Test basic functionality to make sure CuPy works
    try:
        # Try to create a simple array and move it to the GPU
        test_array = cp.array([1, 2, 3])
        # Try a simple operation
        test_result = cp.sum(test_array)
        RAPIDS_AVAILABLE = True
        warnings.warn(f"RAPIDS/GPU libraries (CuPy) found and working properly. GPU acceleration available for coverage calculation. CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    except Exception as e:
        RAPIDS_AVAILABLE = False
        warnings.warn(f"CuPy found but failed to execute basic operations: {e}. GPU acceleration will NOT be available for coverage calculation.")
        cp = None # Define as None if not working
    
    # Check specifically for map_coordinates
    try:
        if RAPIDS_AVAILABLE:
            from cupyx.scipy.ndimage import map_coordinates as cp_map_coordinates
            # Try to create a simple test with map_coordinates
            test_array = cp.zeros((10, 10))
            coords = cp.array([[5, 5], [5, 5]])
            test_result = cp_map_coordinates(test_array, coords, order=1)
            warnings.warn("RAPIDS (CuPy) map_coordinates is working properly for DEM sampling.")
        else:
            cp_map_coordinates = None
    except Exception as e:
        warnings.warn(f"RAPIDS (CuPy) map_coordinates not working: {e}. DEM sampling will use CPU.")
        cp_map_coordinates = None
except ImportError:
    RAPIDS_AVAILABLE = False
    warnings.warn("RAPIDS/GPU libraries (CuPy) not found. GPU acceleration NOT available for coverage calculation.")
    cp = None # Define as None if not available
    cp_map_coordinates = None

# Constants
ANGULO_SETOR = 120  # Standard sector angle in degrees (120° for 3-sector sites)
LIGHT_SPEED = 299792458  # Speed of light in m/s
EARTH_RADIUS_KM = 6371.0 # Earth's radius in km

def calcular_eirp(potencia_watts, ganho_antena):
    """
    Calculate Effective Isotropic Radiated Power (EIRP).
    
    Parameters
    ----------
    potencia_watts : float
        Transmitter power in watts
    ganho_antena : float
        Antenna gain in dBi
        
    Returns
    -------
    float
        EIRP in watts
    """
    # Convert antenna gain from dBi to linear
    ganho_linear = 10 ** (ganho_antena / 10)
    
    # Calculate EIRP
    eirp = potencia_watts * ganho_linear
    
    return eirp

def calcular_raio_cobertura_aprimorado(eirp, freq_mhz, tipo_area='urbana'):
    """
    Calculate coverage radius based on EIRP and frequency.
    
    Parameters
    ----------
    eirp : float
        Effective Isotropic Radiated Power in watts
    freq_mhz : float
        Frequency in MHz
    tipo_area : str, optional
        Area type ('urbana', 'suburbana', 'rural')
        
    Returns
    -------
    float
        Coverage radius in kilometers
    """
    # Convert EIRP to dBm
    eirp_dbm = 10 * math.log10(eirp * 1000)
    
    # Simplified path loss model parameters based on area type
    if tipo_area == 'urbana':
        # Higher path loss in urban areas
        k1, k2, k3 = 120, 35, 0.5
    elif tipo_area == 'suburbana':
        # Medium path loss in suburban areas
        k1, k2, k3 = 110, 30, 0.4
    else:  # rural
        # Lower path loss in rural areas
        k1, k2, k3 = 100, 25, 0.3
    
    # Simplified path loss model based on Okumura-Hata
    # Assumes -90 dBm as minimum received power threshold
    path_loss = eirp_dbm - (-90)  # EIRP - receiver sensitivity
    
    # Calculate radius using the model
    log_r = (path_loss - k1 - k2 * math.log10(freq_mhz)) / (10 * k3)
    raio_km = 10 ** log_r
    
    # Limit maximum radius to a reasonable value
    return min(raio_km, 30)

def calcular_area_cobertura(raio, angulo=ANGULO_SETOR):
    """
    Calculate coverage area based on radius and sector angle.
    
    Parameters
    ----------
    raio : float
        Coverage radius in kilometers
    angulo : float, optional
        Sector angle in degrees
        
    Returns
    -------
    float
        Coverage area in square kilometers
    """
    # Convert radius to meters for calculation
    raio_m = raio * 1000
    
    # For full circle (360°)
    if angulo >= 360:
        return math.pi * (raio_m ** 2) / 1000000  # Convert back to km²
    
    # For sectors
    area_m2 = (math.pi * (raio_m ** 2) * angulo) / 360
    return area_m2 / 1000000  # Convert to km²

def criar_setor_preciso(lat, lon, raio, azimute, angulo=ANGULO_SETOR, resolucao=30):
    """
    Create a precise sector polygon for a cellular antenna.
    
    Parameters
    ----------
    lat : float
        Latitude of the base station
    lon : float
        Longitude of the base station
    raio : float
        Coverage radius in kilometers
    azimute : float
        Antenna azimuth in degrees (0-360)
    angulo : float, optional
        Sector beam width in degrees
    resolucao : int, optional
        Number of points to use for the sector arc
        
    Returns
    -------
    shapely.geometry.Polygon
        Polygon representing the coverage sector
    """
    # Convert radius to meters
    raio_m = raio * 1000
    
    # Calculate the start and end angles of the sector
    start_angle = (azimute - angulo / 2) % 360
    end_angle = (azimute + angulo / 2) % 360
    
    # Convert angles to radians
    start_angle_rad = math.radians(start_angle)
    end_angle_rad = math.radians(end_angle)
    
    # Handle the case where the end angle is less than the start angle
    if end_angle < start_angle:
        end_angle_rad += 2 * math.pi
    
    # Create a point at the origin
    origin = Point(lon, lat)
    
    # Create points for the sector
    points = [origin]
    
    # Add points along the arc
    angle_step = math.radians(angulo) / resolucao
    for i in range(resolucao + 1):
        angle = start_angle_rad + i * angle_step
        dx = raio_m * math.sin(angle) / 111320  # Approximate conversion to degrees longitude
        dy = raio_m * math.cos(angle) / 110540  # Approximate conversion to degrees latitude
        points.append(Point(lon + dx, lat + dy))
    
    # Close the polygon
    points.append(origin)
    
    # Create the polygon from the points
    coords = [(p.x, p.y) for p in points]
    return Polygon(coords)

def calcular_ponto_geodesico(lon, lat, azimute, distancia_km):
    """
    Calculate a new geodetic point given a starting point, azimuth and distance.
    
    Parameters
    ----------
    lon : float
        Longitude of the starting point
    lat : float
        Latitude of the starting point
    azimute : float
        Azimuth in degrees (0-360)
    distancia_km : float
        Distance in kilometers
        
    Returns
    -------
    tuple
        (new_lon, new_lat)
    """
    # Earth's radius in km
    R = 6371.0
    
    # Convert to radians
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    azimute_rad = math.radians(azimute)
    
    # Angular distance
    d_angular = distancia_km / R
    
    # Calculate new point
    lat2 = math.asin(math.sin(lat1) * math.cos(d_angular) + 
                     math.cos(lat1) * math.sin(d_angular) * math.cos(azimute_rad))
    
    lon2 = lon1 + math.atan2(math.sin(azimute_rad) * math.sin(d_angular) * math.cos(lat1),
                            math.cos(d_angular) - math.sin(lat1) * math.sin(lat2))
    
    # Convert back to degrees
    return (math.degrees(lon2), math.degrees(lat2))

def calcular_raio_frenel(freq_mhz, distancia_total_km, n=1):
    """
    Calculate the radius of a Fresnel zone at a given distance.
    
    Parameters
    ----------
    freq_mhz : float
        Frequency in MHz
    distancia_total_km : float
        Total distance between transmitter and receiver in kilometers
    n : int, optional
        Fresnel zone number (usually 1)
        
    Returns
    -------
    float
        Radius of the Fresnel zone in meters
    """
    # Convert frequency to Hz and distance to meters
    freq_hz = freq_mhz * 1000000
    distancia_m = distancia_total_km * 1000
    
    # Calculate the wavelength
    lambda_m = LIGHT_SPEED / freq_hz
    
    # Fresnel zone radius assuming equal distances from point
    radius_m = math.sqrt((n * lambda_m * distancia_m) / 4)
    
    return radius_m

# --- Vectorized GPU Helper Functions ---

def _calcular_pontos_geodesicos_gpu(lon_orig, lat_orig, azimutes_gpu, distancias_gpu):
    """Vectorized calculation of geodetic points using CuPy (spherical approx)."""
    xp = cp
    R = EARTH_RADIUS_KM

    # Ensure inputs are cupy arrays
    if not isinstance(azimutes_gpu, xp.ndarray): azimutes_gpu = xp.asarray(azimutes_gpu)
    if not isinstance(distancias_gpu, xp.ndarray): distancias_gpu = xp.asarray(distancias_gpu)

    # Convert scalars to radians
    lat1 = xp.radians(lat_orig)
    lon1 = xp.radians(lon_orig)

    # Convert input arrays to radians
    azimutes_rad = xp.radians(azimutes_gpu) # Shape (num_angulos,)
    # Reshape distances for broadcasting: (1, num_passos)
    distancias_km_bc = distancias_gpu.reshape(1, -1)
    d_angular = distancias_km_bc / R # Shape (1, num_passos)

    # Reshape azimutes for broadcasting: (num_angulos, 1)
    azimutes_rad_bc = azimutes_rad.reshape(-1, 1)

    # Calculate new latitudes (lat2) using broadcasting
    # Result shape: (num_angulos, num_passos)
    sin_lat1 = xp.sin(lat1)
    cos_lat1 = xp.cos(lat1)
    sin_d_angular = xp.sin(d_angular)
    cos_d_angular = xp.cos(d_angular)
    cos_azimutes = xp.cos(azimutes_rad_bc)

    lat2 = xp.arcsin(sin_lat1 * cos_d_angular +
                     cos_lat1 * sin_d_angular * cos_azimutes)

    # Calculate new longitudes (lon2) using broadcasting
    sin_azimutes = xp.sin(azimutes_rad_bc)
    sin_lat2 = xp.sin(lat2)

    lon2 = lon1 + xp.arctan2(sin_azimutes * sin_d_angular * cos_lat1,
                            cos_d_angular - sin_lat1 * sin_lat2)

    # Convert back to degrees
    return xp.degrees(lon2), xp.degrees(lat2)

def _calcular_raio_frenel_gpu(freq_mhz, distancias_km_gpu, n=1):
    """Vectorized calculation of Fresnel zone radius using CuPy."""
    xp = cp
    if not isinstance(distancias_km_gpu, xp.ndarray):
         distancias_km_gpu = xp.asarray(distancias_km_gpu)

    freq_hz = freq_mhz * 1000000
    distancias_m = distancias_km_gpu * 1000
    if freq_hz == 0: return xp.zeros_like(distancias_m)
    lambda_m = LIGHT_SPEED / freq_hz
    # Ensure positive values under sqrt, use abs
    radius_m = xp.sqrt(abs(n * lambda_m * distancias_m) / 4)
    return radius_m

def _sample_dem_gpu(lons_gpu, lats_gpu, dem_data_gpu, transform):
    """Sample DEM data on GPU using bilinear interpolation."""
    xp = cp
    if not cp_map_coordinates:
         raise ImportError("cupyx.scipy.ndimage.map_coordinates is required for GPU DEM sampling.")

    # Get inverse transform to convert world (lon, lat) to pixel (col, row)
    try:
        inv_transform = ~transform
    except Exception as e:
        raise ValueError(f"Could not invert affine transform: {e}")

    # Apply inverse transform to world coordinates
    # Input shapes: (num_angulos, num_passos)
    # Output shapes: (num_angulos, num_passos)
    cols_gpu, rows_gpu = inv_transform * (lons_gpu, lats_gpu)

    # map_coordinates expects coordinates as (ndim, npoints) array
    # Our coordinates are (num_angulos, num_passos) for rows and cols
    # We need to flatten them and stack them
    coords_gpu = xp.vstack((rows_gpu.ravel(), cols_gpu.ravel())) # Shape (2, num_angulos * num_passos)

    # Perform bilinear interpolation (order=1)
    # Use mode='nearest' to handle out-of-bounds coordinates gracefully
    sampled_elevations_flat = cp_map_coordinates(dem_data_gpu, coords_gpu, order=1, mode='nearest', cval=xp.nan)

    # Reshape back to (num_angulos, num_passos)
    sampled_elevations_gpu = sampled_elevations_flat.reshape(lons_gpu.shape)

    return sampled_elevations_gpu


# --- Main Calculation Function ---

def calc_effective_radius(lat, lon, azimuth, dem_model, original_radius_km, 
                          angulo_setor=ANGULO_SETOR, num_angulos=16, num_passos=50,
                          considerar_difracao=True, considerar_reflexao=True, # Simplified reflection
                          freq_mhz=900, # Add frequency for Fresnel calc
                          use_gpu=False):
    """
    Calculate effective coverage radius considering terrain profile.
    Uses vectorized CuPy operations if use_gpu=True.
    
    Parameters
    ----------
    lat : float
        Latitude of the base station
    lon : float
        Longitude of the base station
    azimuth : float
        Antenna azimuth in degrees
    dem_model : TerrainModel
        Digital Elevation Model object. Must provide access to DEM data
        as `dem_model.dem_data` (numpy array) and `dem_model.transform` (affine).
    original_radius_km : float
        Original theoretical coverage radius in kilometers
    angulo_setor : float, optional
        Sector beam width in degrees
    num_angulos : int, optional
        Number of angles to sample within the sector
    num_passos : int, optional
        Number of steps along each angle to sample
    considerar_difracao : bool, optional
        Whether to consider simple diffraction effects (Fresnel zone)
    considerar_reflexao : bool, optional
        Whether to consider simple reflection effects (flat surfaces)
    freq_mhz : float, optional
        Frequency in MHz (used for Fresnel calculation)
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False)
        
    Returns
    ------
    dict
        Dictionary containing:
        - 'effective_radii': dict {angle: radius_km}
        - 'profiles': dict {angle: {'distances': [km], 'elevations': [m], 'visible': [bool]}} (CPU only)
                       or None if use_gpu=True (profiles not generated in GPU path for efficiency)
        - 'original_radius': float (km)
    """
    if not isinstance(dem_model, TerrainModel):
        raise ValueError("dem_model must be a TerrainModel instance")
        
    # --- GPU Path ---
    if use_gpu and RAPIDS_AVAILABLE:
        warnings.warn("Calculating effective radius on GPU using CuPy.")
        xp = cp
        try:
            # 1. Get DEM data and transform, move DEM to GPU
            dem_data_cpu = dem_model.dem_data
            transform = dem_model.transform
            if dem_data_cpu is None or transform is None:
                 raise ValueError("dem_model must provide .dem_data (numpy array) and .transform (affine)")
            dem_data_gpu = xp.asarray(dem_data_cpu, dtype=xp.float32) # Use float32

            # 2. Get base station elevation on GPU
            base_elev_gpu = _sample_dem_gpu(xp.array([lon]), xp.array([lat]), dem_data_gpu, transform)
            base_elev = base_elev_gpu.item() # Get scalar value back to CPU
            if xp.isnan(base_elev):
                warnings.warn("Base station coordinates are outside DEM bounds or DEM has NaN at location.")
                # Return empty results or default radius? Returning default for now.
                return {"effective_radii": {a: original_radius_km for a in np.linspace(start_angle, end_angle, num_angulos)},
                        "profiles": None, "original_radius": original_radius_km}

            # 3. Generate angles and distances arrays on GPU
            start_angle = (azimuth - angulo_setor / 2) % 360
            end_angle = (azimuth + angulo_setor / 2) % 360
            # Ensure angles wrap correctly if crossing 360
            gpu_angles = xp.linspace(start_angle, end_angle, num_angulos)
            # Handle wrap-around for linspace across 360->0 boundary
            if end_angle < start_angle:
                 gpu_angles = xp.linspace(start_angle, end_angle + 360, num_angulos) % 360

            # Distances from 0 up to original_radius_km
            gpu_distances = xp.linspace(0, original_radius_km, num_passos + 1)[1:] # Steps 1 to num_passos

            # 4. Calculate all sample point coordinates on GPU
            # Input shapes: scalar, scalar, (num_angulos,), (num_passos,)
            # Output shapes: (num_angulos, num_passos)
            lons_gpu, lats_gpu = _calcular_pontos_geodesicos_gpu(lon, lat, gpu_angles, gpu_distances)

            # 5. Sample DEM elevations for all points on GPU
            # Output shape: (num_angulos, num_passos)
            elevations_gpu = _sample_dem_gpu(lons_gpu, lats_gpu, dem_data_gpu, transform)

            # 6. Calculate Line-of-Sight (LoS) and Visibility on GPU
            # Reshape distances for broadcasting: (1, num_passos)
            dist_bc = gpu_distances.reshape(1, -1)

            # Adjust for Earth's curvature (simplified)
            # earth_curve_correction = dist_bc**2 / (2 * EARTH_RADIUS_KM) # Correction in km
            # Convert elevation difference to km for comparison (or curve to m)
            # Let's work in meters:
            earth_curve_correction_m = (dist_bc * 1000)**2 / (2 * EARTH_RADIUS_KM * 1000) # Correction in meters

            # Calculate elevation difference relative to base station
            # Output shape: (num_angulos, num_passos)
            delta_elev_gpu = elevations_gpu - base_elev

            # Calculate expected elevation of the line-of-sight ray at each point
            # Line from (0, base_elev) to (dist, elev_at_dist)
            # Slope (m/km): delta_elev_gpu / dist_bc (where dist_bc > 0)
            # Avoid division by zero for first step (dist=0 implicitly visible)
            # LoS elevation = base_elev + slope * distance
            # Simplified check: is point_elevation <= base_elev + (delta_elev_at_max_dist / max_dist) * current_dist ?
            # A more standard LoS check: is the point below the line connecting base to previous points?
            # Let's use the direct line from base to point:
            # Required elevation of LoS ray at point = base_elev + earth_curve_correction_m
            los_req_elev_gpu = base_elev + earth_curve_correction_m

            # Initial visibility: point elevation must be <= LoS requirement
            # Shape: (num_angulos, num_passos)
            visible_gpu = elevations_gpu <= los_req_elev_gpu

            # Check for obstructions between base and point (more robust LoS)
            # For each point (a, p), check if any previous point (a, k<p) obstructs the view
            # Obstruction if elev[k] > base_elev + (elev[p]-base_elev)*(dist[k]/dist[p]) + curve_correction[k]
            # This iterative check is hard to vectorize directly. Let's stick to the simple LoS for now.
            # TODO: Implement a vectorized version of the iterative LoS check if needed.


            # 7. Apply Diffraction / Reflection modifications (Simplified)
            if considerar_difracao:
                fresnel_radius_m = _calcular_raio_frenel_gpu(freq_mhz, dist_bc) # Shape (1, num_passos)
                # Clearance needed (in meters)
                clearance_gpu = los_req_elev_gpu - elevations_gpu # How much LoS is above terrain
                # If not visible initially, check if clearance > -0.6 * Fresnel radius
                visible_gpu = xp.logical_or(visible_gpu, clearance_gpu > -0.6 * fresnel_radius_m)

            if considerar_reflexao:
                 # Simple check for flat terrain between steps (hard to vectorize perfectly)
                 # Approximation: Check slope between consecutive points
                 if elevations_gpu.shape[1] > 1: # Need at least 2 points per angle
                      delta_dist_km = original_radius_km / num_passos
                      # Slopes between consecutive points along each angle
                      slopes_gpu = xp.diff(elevations_gpu, axis=1) / (delta_dist_km * 1000) # Slope m/m
                      # Pad slopes array to match visibility shape
                      slopes_padded_gpu = xp.pad(slopes_gpu, ((0,0),(0,1)), mode='edge')
                      # Check if slope is near flat (< 5%) AND point wasn't already visible
                      potentially_reflective_gpu = xp.abs(slopes_padded_gpu) < 0.05
                      visible_gpu = xp.logical_or(visible_gpu, potentially_reflective_gpu)


            # 8. Determine effective radius for each angle
            # Find the index of the *last* visible step along each angle (axis=1)
            # We need indices where visible_gpu is True.
            visible_indices = xp.argwhere(visible_gpu) # Gets [[row, col], ...] pairs where True

            # Find the maximum column index (step index) for each row (angle index)
            effective_step_indices = xp.zeros(num_angulos, dtype=xp.int32) - 1 # Initialize with -1
            if visible_indices.size > 0:
                 # Use scatter_max (or equivalent logic) to find max column index per row
                 # Alternative: sort by row, then find max col per row block (complex)
                 # Simpler CPU fallback for index finding (might be faster than complex GPU logic):
                 vis_indices_cpu = visible_indices.get()
                 eff_indices_cpu = np.full(num_angulos, -1, dtype=np.int32)
                 for r, c in vis_indices_cpu:
                      eff_indices_cpu[r] = max(eff_indices_cpu[r], c)
                 effective_step_indices = xp.asarray(eff_indices_cpu) # Move back to GPU if needed later


            # Get the corresponding distance for the last visible step index
            # Handle case where no step was visible (index remains -1)
            # gpu_distances shape is (num_passos,)
            effective_radii_gpu = xp.where(effective_step_indices >= 0,
                                           gpu_distances[effective_step_indices],
                                           0) # Radius is 0 if no step visible


            # 9. Format results (transfer back to CPU)
            angles_cpu = gpu_angles.get()
            effective_radii_cpu = effective_radii_gpu.get()

            final_radii = dict(zip(angles_cpu, effective_radii_cpu))

            # Profiles are not generated in GPU mode for efficiency
            final_profiles = None

            return {
                "effective_radii": final_radii,
                "profiles": final_profiles,
                "original_radius": original_radius_km
            }

        except Exception as e:
            warnings.warn(f"GPU effective radius calculation failed: {e}. Falling back to CPU.")
            # Fallback handled by the structure below


    # --- CPU Path ---
    warnings.warn("Calculating effective radius on CPU.")
    xp = np # Use numpy for CPU path

    # Get base station elevation (CPU)
    try:
        base_elev = dem_model.get_elevation(lon, lat)
        if base_elev is None or np.isnan(base_elev):
            warnings.warn("Base station coordinates are outside DEM bounds or DEM has NaN at location.")
            # Generate angles for the dictionary keys even if returning default
            start_angle = (azimuth - angulo_setor / 2) % 360
            end_angle = (azimuth + angulo_setor / 2) % 360
            cpu_angles = np.linspace(start_angle, end_angle, num_angulos)
            if end_angle < start_angle:
                cpu_angles = np.linspace(start_angle, end_angle + 360, num_angulos) % 360
            return {"effective_radii": {a: original_radius_km for a in cpu_angles},
                    "profiles": {}, "original_radius": original_radius_km}
    except Exception as e:
         warnings.warn(f"Failed to get base elevation: {e}. Returning default radius.")
         start_angle = (azimuth - angulo_setor / 2) % 360
         end_angle = (azimuth + angulo_setor / 2) % 360
         cpu_angles = np.linspace(start_angle, end_angle, num_angulos)
         if end_angle < start_angle:
              cpu_angles = np.linspace(start_angle, end_angle + 360, num_angulos) % 360
         return {"effective_radii": {a: original_radius_km for a in cpu_angles},
                     "profiles": {}, "original_radius": original_radius_km}

        
    # Define angles to sample within the sector
    start_angle = (azimuth - angulo_setor / 2) % 360
    end_angle = (azimuth + angulo_setor / 2) % 360
    cpu_angles = np.linspace(start_angle, end_angle, num_angulos)
    # Handle wrap-around for linspace across 360->0 boundary
    if end_angle < start_angle:
         cpu_angles = np.linspace(start_angle, end_angle + 360, num_angulos) % 360
    
    # Initialize results
    effective_radii = {}
    profiles = {}
    
    # Sample along each angle
    for current_angle in cpu_angles:
        # Sample points along this angle
        distance_step = original_radius_km / num_passos
        distances = []
        elevations = []
        visible = []
        last_valid_elevation = base_elev # Keep track of last known elevation
        
        # Line of sight algorithm
        for j in range(1, num_passos + 1):
            current_distance = j * distance_step
            punto_lon, punto_lat = calcular_ponto_geodesico(lon, lat, current_angle, current_distance)
            
            # Get elevation at this point (CPU)
            try:
                punto_elev = dem_model.get_elevation(punto_lon, punto_lat)
            except Exception as e:
                 warnings.warn(f"Error getting elevation for point ({punto_lon},{punto_lat}) at angle {current_angle}: {e}")
                 punto_elev = None # Treat as end of valid profile

            if punto_elev is None or np.isnan(punto_elev):
                # If point is outside DEM or NaN, stop profile for this angle
                # Use the last valid elevation for calculation up to this point
                punto_elev = last_valid_elevation # Use last known for final check
                # We break *after* potentially marking the last valid point as visible/not
                distances.append(current_distance)
                elevations.append(punto_elev) # Append last valid or None? Append last valid for profile consistency.
                # Visibility check for this last point
                if current_distance > 0 and len(elevations)>1:
                    earth_curve_correction_m = (current_distance * 1000)**2 / (2 * EARTH_RADIUS_KM * 1000)
                    los_req_elev = base_elev + earth_curve_correction_m
                    is_visible = punto_elev <= los_req_elev # Simple check
                    # TODO: Implement iterative LoS check here if needed for CPU path accuracy
                    # Apply Diffraction/Reflection for CPU path
                    if not is_visible and considerar_difracao:
                         fresnel_radius = calcular_raio_frenel(freq_mhz, current_distance)
                         clearance = los_req_elev - punto_elev
                         is_visible = clearance > -0.6 * fresnel_radius
                    if not is_visible and considerar_reflexao:
                          if j > 1: # Need previous slope
                               prev_slope = (elevations[-2] - elevations[-1]) / (distance_step * 1000) if distance_step > 0 else 0
                               if abs(prev_slope) < 0.05: is_visible = True
                    visible.append(is_visible)
                else:
                    visible.append(True) # First point is always visible from base

                break # Stop processing this angle's profile

            last_valid_elevation = punto_elev # Update last known good elevation
            distances.append(current_distance)
            elevations.append(punto_elev)
            
            # Calculate line of sight (CPU)
            if current_distance > 0:
                earth_curve_correction_m = (current_distance * 1000)**2 / (2 * EARTH_RADIUS_KM * 1000)
                los_req_elev = base_elev + earth_curve_correction_m
                is_visible = punto_elev <= los_req_elev # Simple check
                # TODO: Implement iterative LoS check here if needed for CPU path accuracy

                # Apply Diffraction/Reflection for CPU path
                if not is_visible and considerar_difracao:
                     fresnel_radius = calcular_raio_frenel(freq_mhz, current_distance)
                     clearance = los_req_elev - punto_elev
                     is_visible = clearance > -0.6 * fresnel_radius
                if not is_visible and considerar_reflexao:
                     if j > 1: # Need previous slope
                           prev_slope = (elevations[-2] - elevations[-1]) / (distance_step * 1000) if distance_step > 0 else 0
                           if abs(prev_slope) < 0.05: is_visible = True
                
                visible.append(is_visible)
            else:
                visible.append(True) # Point at base station is visible

        # Determine effective radius (last distance where visible)
        effective_radius = 0
        if distances: # Check if we actually processed any steps
            last_visible_idx = -1
            for idx in range(len(visible) - 1, -1, -1):
                 if visible[idx]:
                      last_visible_idx = idx
                      break
            if last_visible_idx >= 0:
                effective_radius = distances[last_visible_idx]
            
        # Store results
        effective_radii[current_angle] = effective_radius
        profiles[current_angle] = {
            "distances": distances,
            "elevations": elevations,
            "visible": visible
        }
    
    # Return the results (CPU path)
    return {
        "effective_radii": effective_radii,
        "profiles": profiles,
        "original_radius": original_radius_km
    }


def amostra_segura_mde(dem_model, lon, lat, valor_default=None):
    """Safely sample elevation from a DEM at given coordinates (CPU)."""
    # This remains a CPU function, potentially called by CPU path of calc_effective_radius
    # or other modules.
    try:
        if not isinstance(dem_model, TerrainModel):
             raise ValueError("dem_model must be a TerrainModel instance")
        elevation = dem_model.get_elevation(lon, lat)
        # Check for NaN explicitly
        return elevation if elevation is not None and not np.isnan(elevation) else valor_default
    except Exception:
        return valor_default 

def calculate_coverage_radius(eirp_dbm, freq_mhz, loss_exponent=3.5, receiver_sensitivity_dbm=-90):
    """
    Calculate the theoretical coverage radius in meters using the log-distance path loss model.
    
    Parameters
    ----------
    eirp_dbm : float
        Effective Isotropic Radiated Power in dBm
    freq_mhz : float
        Frequency in MHz
    loss_exponent : float, optional
        Path loss exponent (typically 2.0 to 5.0)
    receiver_sensitivity_dbm : float, optional
        Minimum received power threshold in dBm
        
    Returns
    -------
    float
        Coverage radius in meters
    """
    # Constants
    # Free space path loss at 1m reference distance
    wavelength = 3e8 / (freq_mhz * 1e6)  # wavelength in meters
    fspl_1m = 20 * np.log10(4 * np.pi / wavelength)
    
    # Calculate maximum allowed path loss
    max_path_loss = eirp_dbm - receiver_sensitivity_dbm - fspl_1m
    
    # Calculate radius
    radius_m = 10 ** (max_path_loss / (10 * loss_exponent))
    
    # Limit to a reasonable value (30 km)
    return min(radius_m, 30000)

def calculate_signal_strength(eirp_dbm, freq_mhz, distance_m, loss_exponent=3.5):
    """
    Calculate the signal strength at a specific distance using the log-distance path loss model.
    
    Parameters
    ----------
    eirp_dbm : float
        Effective Isotropic Radiated Power in dBm
    freq_mhz : float
        Frequency in MHz
    distance_m : float
        Distance from transmitter in meters
    loss_exponent : float, optional
        Path loss exponent (typically 2.0 to 5.0)
        
    Returns
    -------
    float
        Received signal strength in dBm
    """
    # Constants
    # Free space path loss at 1m reference distance
    wavelength = 3e8 / (freq_mhz * 1e6)  # wavelength in meters
    fspl_1m = 20 * np.log10(4 * np.pi / wavelength)
    
    # Calculate path loss at the given distance
    path_loss = fspl_1m + 10 * loss_exponent * np.log10(distance_m)
    
    # Calculate received signal strength
    received_power_dbm = eirp_dbm - path_loss
    
    return received_power_dbm 