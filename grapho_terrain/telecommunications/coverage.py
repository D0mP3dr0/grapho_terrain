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

# Constants
ANGULO_SETOR = 120  # Standard sector angle in degrees (120° for 3-sector sites)
LIGHT_SPEED = 299792458  # Speed of light in m/s

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

def calc_effective_radius(lat, lon, azimuth, dem_model, original_radius_km, 
                          angulo_setor=ANGULO_SETOR, num_angulos=16, num_passos=50,
                          considerar_difracao=True, considerar_reflexao=True):
    """
    Calculate effective coverage radius considering terrain profile.
    
    Parameters
    ----------
    lat : float
        Latitude of the base station
    lon : float
        Longitude of the base station
    azimuth : float
        Antenna azimuth in degrees
    dem_model : TerrainModel
        Digital Elevation Model
    original_radius_km : float
        Original theoretical coverage radius in kilometers
    angulo_setor : float, optional
        Sector beam width in degrees
    num_angulos : int, optional
        Number of angles to sample within the sector
    num_passos : int, optional
        Number of steps along each angle to sample
    considerar_difracao : bool, optional
        Whether to consider diffraction effects
    considerar_reflexao : bool, optional
        Whether to consider reflection effects
        
    Returns
    -------
    dict
        Dictionary with effective radius for each sampled angle
    """
    if not isinstance(dem_model, TerrainModel):
        raise ValueError("dem_model must be a TerrainModel instance")
        
    # Get base station elevation
    base_elev = dem_model.get_elevation(lon, lat)
    if base_elev is None:
        return {"effective_radius": original_radius_km, "profile": []}
        
    # Define angles to sample within the sector
    start_angle = (azimuth - angulo_setor / 2) % 360
    angle_step = angulo_setor / (num_angulos - 1) if num_angulos > 1 else 0
    
    # Initialize results
    effective_radii = {}
    profiles = {}
    
    # Sample along each angle
    for i in range(num_angulos):
        current_angle = (start_angle + i * angle_step) % 360
        
        # Sample points along this angle
        distance_step = original_radius_km / num_passos
        distances = []
        elevations = []
        visible = []
        
        # Line of sight algorithm
        for j in range(1, num_passos + 1):
            # Calculate distance for this step
            current_distance = j * distance_step
            
            # Calculate geographic coordinates
            punto_lon, punto_lat = calcular_ponto_geodesico(lon, lat, current_angle, current_distance)
            
            # Get elevation at this point
            punto_elev = dem_model.get_elevation(punto_lon, punto_lat)
            
            if punto_elev is None:
                # If point is outside DEM, use last valid point
                break
                
            # Store distance and elevation
            distances.append(current_distance)
            elevations.append(punto_elev)
            
            # Calculate line of sight
            # Line equation: y = m*x + b
            if current_distance > 0:
                # Adjust for Earth's curvature (simplified)
                earth_curve_correction = current_distance**2 / (2 * 6371)  # 6371 km is Earth's radius
                
                # Calculate line of sight elevation at this distance
                m = (punto_elev - base_elev) / current_distance
                line_of_sight = base_elev + m * current_distance
                
                # Check if terrain blocks line of sight
                is_visible = punto_elev <= line_of_sight + earth_curve_correction
                
                # Consider diffraction if enabled
                if not is_visible and considerar_difracao:
                    # Simplified Fresnel zone consideration
                    # A rule of thumb: if obstacle is within 0.6 of first Fresnel zone, signal is significantly attenuated
                    # We'll use a simple approximation here
                    fresnel_radius = calcular_raio_frenel(900, current_distance)  # Assuming 900 MHz
                    fresnel_clearance = line_of_sight - punto_elev
                    is_visible = fresnel_clearance > -0.6 * fresnel_radius
                
                # Consider reflection if enabled
                if not is_visible and considerar_reflexao:
                    # Simplified reflection model
                    # Check if there are flat surfaces that could cause reflection
                    if j > 1 and j < num_passos:
                        prev_slope = (elevations[-2] - elevations[-1]) / distance_step
                        if abs(prev_slope) < 0.05:  # Less than 5% slope could cause reflection
                            is_visible = True
                
                visible.append(is_visible)
            else:
                visible.append(True)
        
        # Determine effective radius (last point where signal is visible)
        if len(visible) > 0:
            last_visible_idx = len(visible) - 1
            while last_visible_idx >= 0 and not visible[last_visible_idx]:
                last_visible_idx -= 1
                
            if last_visible_idx >= 0:
                effective_radius = distances[last_visible_idx]
            else:
                effective_radius = 0
        else:
            effective_radius = 0
            
        # Store results
        effective_radii[current_angle] = effective_radius
        profiles[current_angle] = {
            "distances": distances,
            "elevations": elevations,
            "visible": visible
        }
    
    # Return the results
    return {
        "effective_radii": effective_radii,
        "profiles": profiles,
        "original_radius": original_radius_km
    }

def amostra_segura_mde(dem_model, lon, lat, valor_default=None):
    """
    Safely sample elevation from a DEM at given coordinates.
    
    Parameters
    ----------
    dem_model : TerrainModel
        Digital Elevation Model
    lon : float
        Longitude
    lat : float
        Latitude
    valor_default : float, optional
        Default value to return if sampling is not possible
        
    Returns
    -------
    float
        Elevation at the given point or default value
    """
    try:
        elevation = dem_model.get_elevation(lon, lat)
        return elevation if elevation is not None else valor_default
    except Exception:
        return valor_default 