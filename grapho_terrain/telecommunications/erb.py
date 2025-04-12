"""
Cellular base station (ERB) data processing and analysis.

This module provides functions for loading, processing, and analyzing
cellular network base station (ERB) data.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import os
from ..core.data_model import UrbanFeature
from .coverage import calcular_eirp, calcular_raio_cobertura_aprimorado, criar_setor_preciso

class ERB(UrbanFeature):
    """
    Cellular network base station (ERB) representation.
    """
    
    def __init__(self, erb_id, geometry, operadora=None, tecnologia=None, freq_mhz=None, 
                 potencia_watts=None, ganho_antena=None, azimute=None, 
                 altura_m=None, attributes=None, crs=None):
        """
        Initialize an ERB.
        
        Parameters
        ----------
        erb_id : str or int
            Unique identifier for the ERB
        geometry : shapely.geometry
            Point geometry of the ERB location
        operadora : str, optional
            Operator name
        tecnologia : str, optional
            Technology (e.g., '4G', '5G')
        freq_mhz : float, optional
            Frequency in MHz
        potencia_watts : float, optional
            Transmit power in watts
        ganho_antena : float, optional
            Antenna gain in dBi
        azimute : float, optional
            Antenna azimuth in degrees
        altura_m : float, optional
            Antenna height in meters
        attributes : dict, optional
            Dictionary of additional attributes
        crs : str or dict, optional
            Coordinate reference system
        """
        # Initialize the base class
        super().__init__(erb_id, geometry, "erb", attributes, crs)
        
        # Add ERB-specific attributes
        self.operadora = operadora
        self.tecnologia = tecnologia
        self.freq_mhz = freq_mhz
        self.potencia_watts = potencia_watts
        self.ganho_antena = ganho_antena
        self.azimute = azimute
        self.altura_m = altura_m
        
        # Add ERB-specific attributes to the attributes dictionary
        if operadora is not None:
            self.attributes["operadora"] = operadora
        if tecnologia is not None:
            self.attributes["tecnologia"] = tecnologia
        if freq_mhz is not None:
            self.attributes["freq_mhz"] = freq_mhz
        if potencia_watts is not None:
            self.attributes["potencia_watts"] = potencia_watts
        if ganho_antena is not None:
            self.attributes["ganho_antena"] = ganho_antena
        if azimute is not None:
            self.attributes["azimute"] = azimute
        if altura_m is not None:
            self.attributes["altura_m"] = altura_m
            
    def calculate_eirp(self):
        """
        Calculate the Effective Isotropic Radiated Power (EIRP).
        
        Returns
        -------
        float or None
            EIRP in watts, or None if required attributes are missing
        """
        if self.potencia_watts is not None and self.ganho_antena is not None:
            return calcular_eirp(self.potencia_watts, self.ganho_antena)
        return None
        
    def calculate_coverage_radius(self, tipo_area='urbana'):
        """
        Calculate the theoretical coverage radius.
        
        Parameters
        ----------
        tipo_area : str, optional
            Area type ('urbana', 'suburbana', 'rural')
            
        Returns
        -------
        float or None
            Coverage radius in kilometers, or None if required attributes are missing
        """
        eirp = self.calculate_eirp()
        if eirp is not None and self.freq_mhz is not None:
            return calcular_raio_cobertura_aprimorado(eirp, self.freq_mhz, tipo_area)
        return None
        
    def create_coverage_sector(self, raio=None, tipo_area='urbana', resolucao=30):
        """
        Create a polygon representing the coverage sector.
        
        Parameters
        ----------
        raio : float, optional
            Coverage radius in kilometers (if None, will be calculated)
        tipo_area : str, optional
            Area type ('urbana', 'suburbana', 'rural')
        resolucao : int, optional
            Number of points to use for the sector arc
            
        Returns
        -------
        shapely.geometry.Polygon or None
            Polygon representing the coverage sector, or None if required attributes are missing
        """
        if self.geometry is None or self.azimute is None:
            return None
            
        # Get coordinates
        if self.geometry.geom_type != 'Point':
            return None
            
        lon, lat = self.geometry.x, self.geometry.y
        
        # Get coverage radius
        if raio is None:
            raio = self.calculate_coverage_radius(tipo_area)
            if raio is None:
                return None
                
        # Create the sector polygon
        return criar_setor_preciso(lat, lon, raio, self.azimute, resolucao=resolucao)
        
    def calculate_signal_strength(self, point, loss_exponent=3.5):
        """
        Calculate the signal strength at a specific location.
        
        Parameters
        ----------
        point : shapely.geometry.Point
            The location to calculate signal strength at
        loss_exponent : float, optional
            Path loss exponent (typically 2.0 to 5.0)
            
        Returns
        -------
        float or None
            Received signal strength in dBm, or None if required attributes are missing
        """
        if self.geometry is None or point is None:
            return None
            
        # Calculate EIRP
        eirp = self.calculate_eirp()
        if eirp is None or self.freq_mhz is None:
            return None
            
        # Convert EIRP from watts to dBm
        eirp_dbm = 10 * np.log10(eirp * 1000)
        
        # Calculate distance in meters
        distance_m = self.geometry.distance(point) * 111319  # approximate conversion from degrees to meters
        
        # Use the coverage module to calculate signal strength
        from grapho_terrain.telecommunications.coverage import calculate_signal_strength
        return calculate_signal_strength(eirp_dbm, self.freq_mhz, distance_m, loss_exponent)
        
    def calculate_signal_strength_with_terrain(self, point, dem_model, loss_exponent=3.5, consider_diffraction=True, consider_reflection=True):
        """
        Calculate the signal strength at a specific location considering terrain and obstacles.
        
        Parameters
        ----------
        point : shapely.geometry.Point
            The location to calculate signal strength at
        dem_model : TerrainModel
            Digital Elevation Model object to consider terrain
        loss_exponent : float, optional
            Path loss exponent (typically 2.0 to 5.0)
        consider_diffraction : bool, optional
            Whether to consider diffraction effects around obstacles
        consider_reflection : bool, optional
            Whether to consider reflection from flat surfaces
            
        Returns
        -------
        float or None
            Received signal strength in dBm considering terrain, or None if required attributes are missing
        """
        if self.geometry is None or point is None or dem_model is None:
            return None
            
        # Calculate basic parameters
        eirp = self.calculate_eirp()
        if eirp is None or self.freq_mhz is None or self.azimute is None:
            return None
            
        # Convert EIRP from watts to dBm
        eirp_dbm = 10 * np.log10(eirp * 1000)
        
        # Calculate angle and distance
        dx = point.x - self.geometry.x
        dy = point.y - self.geometry.y
        angle_deg = (np.degrees(np.arctan2(dy, dx)) + 90) % 360  # Convert to azimuth where 0=North
        distance_deg = self.geometry.distance(point)
        distance_km = distance_deg * 111.319  # approximate conversion from degrees to km
        
        # Get base station and point elevations
        from grapho_terrain.telecommunications.coverage import amostra_segura_mde
        erb_elev = amostra_segura_mde(dem_model, self.geometry.x, self.geometry.y, 0)
        if self.altura_m is not None:
            erb_elev += self.altura_m  # Add antenna height
        point_elev = amostra_segura_mde(dem_model, point.x, point.y, 0)
        
        # Check line of sight using effective radius calculation
        from grapho_terrain.telecommunications.coverage import calc_effective_radius
        result = calc_effective_radius(
            self.geometry.y, self.geometry.x, angle_deg, 
            dem_model, distance_km * 1.1,  # Use slightly larger radius than needed
            angulo_setor=30,  # Use a narrow sector for precision
            num_angulos=3,    # Focus on the direct path
            num_passos=50,
            considerar_difracao=consider_diffraction,
            considerar_reflexao=consider_reflection,
            freq_mhz=self.freq_mhz
        )
        
        # Get the effective radius for this angle
        # Find the closest angle in the result
        angles = np.array(list(result["effective_radii"].keys()))
        idx = np.argmin(np.abs(angles - angle_deg))
        closest_angle = angles[idx]
        effective_radius_km = result["effective_radii"][closest_angle]
        
        # Check if the point is within the effective radius
        if distance_km <= effective_radius_km:
            # Calculate signal with terrain effects
            # Apply antenna directivity factor based on angle difference
            azimuth_diff = abs((angle_deg - self.azimute) % 360)
            if azimuth_diff > 180:
                azimuth_diff = 360 - azimuth_diff
                
            # Simple directivity model: -3dB at +/-30 degrees, reduced signal beyond that
            directivity_factor = 0
            if azimuth_diff <= 30:
                directivity_factor = 0  # Full gain within main beam
            elif azimuth_diff <= 90:
                directivity_factor = -3 * (azimuth_diff - 30) / 60  # Linear reduction to -3dB
            else:
                directivity_factor = -10  # Minimal signal outside main coverage
            
            # Calculate basic signal strength
            from grapho_terrain.telecommunications.coverage import calculate_signal_strength
            signal_dbm = calculate_signal_strength(
                eirp_dbm, 
                self.freq_mhz, 
                distance_km * 1000,  # Convert to meters
                loss_exponent
            )
            
            # Apply directivity
            signal_dbm += directivity_factor
            
            return signal_dbm
        else:
            # No line of sight, signal blocked by terrain
            return -120  # Very weak signal (effective no signal)


class ERBLayer:
    """
    A collection of ERB (cellular base station) objects.
    """
    
    def __init__(self, name=None, crs=None):
        """
        Initialize an ERBLayer.
        
        Parameters
        ----------
        name : str, optional
            Name of the layer
        crs : str or dict, optional
            Coordinate reference system
        """
        self.name = name
        self.crs = crs
        self.erbs = {}
        self.metadata = {}
        
    def add_erb(self, erb):
        """
        Add an ERB to the layer.
        
        Parameters
        ----------
        erb : ERB
            ERB to add
        """
        self.erbs[erb.id] = erb
        
        # Update CRS if not already set
        if not self.crs and erb.crs:
            self.crs = erb.crs
            
    def get_erb(self, erb_id):
        """
        Get an ERB by ID.
        
        Parameters
        ----------
        erb_id : str or int
            ID of the ERB
            
        Returns
        -------
        ERB
            The ERB with the given ID
        """
        return self.erbs.get(erb_id)
    
    def to_geodataframe(self):
        """
        Convert the layer to a GeoDataFrame.
        
        Returns
        -------
        GeoDataFrame
            GeoDataFrame containing all ERBs in the layer
        """
        erbs_data = []
        for erb_id, erb in self.erbs.items():
            data = {
                "id": erb_id,
                "geometry": erb.geometry
            }
            data.update(erb.attributes)
            erbs_data.append(data)
            
        gdf = gpd.GeoDataFrame(erbs_data, crs=self.crs)
        return gdf
    
    def create_coverage_sectors(self, tipo_area='urbana'):
        """
        Create coverage sectors for all ERBs.
        
        Parameters
        ----------
        tipo_area : str, optional
            Area type ('urbana', 'suburbana', 'rural')
            
        Returns
        -------
        GeoDataFrame
            GeoDataFrame containing coverage sectors
        """
        sectors_data = []
        
        for erb_id, erb in self.erbs.items():
            sector = erb.create_coverage_sector(tipo_area=tipo_area)
            if sector is not None:
                raio = erb.calculate_coverage_radius(tipo_area)
                data = {
                    "erb_id": erb_id,
                    "geometry": sector,
                    "raio_km": raio
                }
                
                # Include key attributes
                if erb.operadora is not None:
                    data["operadora"] = erb.operadora
                if erb.tecnologia is not None:
                    data["tecnologia"] = erb.tecnologia
                if erb.freq_mhz is not None:
                    data["freq_mhz"] = erb.freq_mhz
                    
                sectors_data.append(data)
                
        sectors_gdf = gpd.GeoDataFrame(sectors_data, crs=self.crs)
        return sectors_gdf
    
    @classmethod
    def from_geodataframe(cls, gdf, name=None):
        """
        Create an ERBLayer from a GeoDataFrame.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame containing ERB data
        name : str, optional
            Name of the layer
            
        Returns
        -------
        ERBLayer
            A new ERBLayer instance
        """
        layer = cls(name=name, crs=gdf.crs)
        
        # Standard column names and their alternate forms
        id_cols = ["id", "erb_id", "ID", "codigo", "CODIGO"]
        operadora_cols = ["operadora", "OPERADORA", "operator", "empresa"]
        tecnologia_cols = ["tecnologia", "TECNOLOGIA", "technology", "tech"]
        freq_cols = ["freq_mhz", "frequencia", "frequencia_mhz", "FREQUENCIA"]
        potencia_cols = ["potencia_watts", "potencia", "POTENCIA", "power"]
        ganho_cols = ["ganho_antena", "ganho", "GANHO", "gain"]
        azimute_cols = ["azimute", "AZIMUTE", "azimuth", "azm"]
        altura_cols = ["altura_m", "altura", "ALTURA", "height"]
        
        # Helper to find a value from a list of possible column names
        def get_value(row, possible_cols, default=None):
            for col in possible_cols:
                if col in row.index and not pd.isna(row[col]):
                    return row[col]
            return default
        
        for _, row in gdf.iterrows():
            # Get the geometry
            geom = row.geometry
            
            # Skip rows without geometry
            if geom is None:
                continue
                
            # Ensure the geometry is a Point
            if geom.geom_type != 'Point':
                continue
                
            # Get the ID
            erb_id = get_value(row, id_cols)
            if erb_id is None:
                erb_id = str(len(layer.erbs))
                
            # Get other attributes
            operadora = get_value(row, operadora_cols)
            tecnologia = get_value(row, tecnologia_cols)
            freq_mhz = get_value(row, freq_cols)
            potencia_watts = get_value(row, potencia_cols)
            ganho_antena = get_value(row, ganho_cols)
            azimute = get_value(row, azimute_cols)
            altura_m = get_value(row, altura_cols)
            
            # Convert numeric values
            if freq_mhz is not None:
                try:
                    freq_mhz = float(freq_mhz)
                except (ValueError, TypeError):
                    freq_mhz = None
                    
            if potencia_watts is not None:
                try:
                    potencia_watts = float(potencia_watts)
                except (ValueError, TypeError):
                    potencia_watts = None
                    
            if ganho_antena is not None:
                try:
                    ganho_antena = float(ganho_antena)
                except (ValueError, TypeError):
                    ganho_antena = None
                    
            if azimute is not None:
                try:
                    azimute = float(azimute)
                except (ValueError, TypeError):
                    azimute = None
                    
            if altura_m is not None:
                try:
                    altura_m = float(altura_m)
                except (ValueError, TypeError):
                    altura_m = None
            
            # Create an ERB object with extracted attributes
            erb = ERB(
                erb_id=erb_id,
                geometry=geom,
                operadora=operadora,
                tecnologia=tecnologia,
                freq_mhz=freq_mhz,
                potencia_watts=potencia_watts,
                ganho_antena=ganho_antena,
                azimute=azimute,
                altura_m=altura_m,
                crs=gdf.crs
            )
            
            # Add the ERB to the layer
            layer.add_erb(erb)
            
        return layer


def carregar_dados_erbs(caminho_csv):
    """
    Load ERB data from a CSV file.
    
    Parameters
    ----------
    caminho_csv : str
        Path to the CSV file
        
    Returns
    -------
    ERBLayer
        Layer containing ERB data
    """
    # Check if file exists
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"File not found: {caminho_csv}")
        
    # Read CSV file
    try:
        df = pd.read_csv(caminho_csv)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
        
    # Check for required columns
    required_cols = ['latitude', 'longitude']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
            
    # Create a GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Create an ERBLayer
    return ERBLayer.from_geodataframe(gdf, name="ERBs from CSV")


def carregar_dados_erbs_personalizado(caminho_csv, crs_origem="EPSG:4326", crs_destino="EPSG:4326", lon_col='longitude', lat_col='latitude'):
    """
    Load ERB data from a CSV file with custom coordinate reference system.
    
    Parameters
    ----------
    caminho_csv : str
        Path to the CSV file
    crs_origem : str, optional
        Original coordinate reference system of the data (default: "EPSG:4326")
    crs_destino : str, optional
        Target coordinate reference system (default: "EPSG:4326")
    lon_col : str, optional
        Column name for longitude data (default: 'longitude')
    lat_col : str, optional
        Column name for latitude data (default: 'latitude')
        
    Returns
    -------
    ERBLayer
        Layer containing ERB data with coordinates in the target CRS
    """
    # Check if file exists
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"File not found: {caminho_csv}")
        
    # Read CSV file
    try:
        df = pd.read_csv(caminho_csv)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
        
    # Check for required columns
    for col in [lon_col, lat_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
            
    # Create a GeoDataFrame with the original CRS
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs_origem)
    
    # Transform to target CRS if different
    if crs_origem != crs_destino:
        try:
            gdf = gdf.to_crs(crs_destino)
            print(f"Successfully transformed coordinates from {crs_origem} to {crs_destino}")
        except Exception as e:
            raise ValueError(f"Error transforming coordinates: {e}")
    
    # Create an ERBLayer
    return ERBLayer.from_geodataframe(gdf, name=f"ERBs from CSV ({crs_destino})")


def calcular_densidade_erb(ponto, gdf_erb, raio=0.01):
    """
    Calculate the ERB density around a point.
    
    Parameters
    ----------
    ponto : shapely.geometry.Point
        Center point
    gdf_erb : GeoDataFrame
        GeoDataFrame containing ERB points
    raio : float, optional
        Radius in decimal degrees
        
    Returns
    -------
    float
        ERB density (count per km²)
    """
    # Create a buffer around the point
    buffer = ponto.buffer(raio)
    
    # Count ERBs within the buffer
    erb_count = sum(gdf_erb.geometry.intersects(buffer))
    
    # Calculate the buffer area (approximate conversion to km²)
    # This is a rough approximation and should be improved for production use
    # by properly calculating the area in square kilometers
    lat = ponto.y
    km_per_degree_lat = 111.32  # km per degree of latitude
    km_per_degree_lon = 111.32 * np.cos(np.radians(lat))  # km per degree of longitude
    
    # Average km per degree
    km_per_degree = (km_per_degree_lat + km_per_degree_lon) / 2
    
    # Convert the buffer area to km²
    area_km2 = (raio ** 2) * np.pi * (km_per_degree ** 2)
    
    # Calculate density
    if area_km2 > 0:
        densidade = erb_count / area_km2
    else:
        densidade = 0
        
    return densidade 

def create_erb_layer(data, id_col='id', lat_col='latitude', lon_col='longitude', crs="EPSG:4326", **kwargs):
    """
    Create an ERB layer from a pandas DataFrame or a list of ERB objects.
    
    Parameters
    ----------
    data : pandas.DataFrame or list of ERB
        Data containing ERB information or list of ERB objects
    id_col : str, optional
        Column name for the ERB ID (for DataFrame input)
    lat_col : str, optional
        Column name for latitude (for DataFrame input)
    lon_col : str, optional
        Column name for longitude (for DataFrame input)
    crs : str or dict, optional
        Coordinate reference system
    **kwargs : dict
        Additional parameters for ERB creation (for DataFrame input)
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing ERB data
    """
    # Handle cases where data is a list of ERB objects
    if isinstance(data, list) and all(isinstance(item, ERB) for item in data):
        # Extract data from ERB objects
        erb_data = []
        for erb in data:
            row_data = {"id": erb.id, "geometry": erb.geometry}
            row_data.update(erb.attributes)
            erb_data.append(row_data)
            
        # Create GeoDataFrame
        return gpd.GeoDataFrame(erb_data, crs=crs)
        
    # Handle DataFrames
    elif isinstance(data, pd.DataFrame):
        # Create points from lat/lon
        if lat_col in data.columns and lon_col in data.columns:
            geometry = [Point(lon, lat) for lon, lat in zip(data[lon_col], data[lat_col])]
        else:
            raise ValueError(f"Columns {lat_col} and/or {lon_col} not found in DataFrame")
            
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
        
        # Create ERB layer
        layer = ERBLayer.from_geodataframe(gdf)
        
        # Return as GeoDataFrame
        return layer.to_geodataframe()
        
    else:
        raise ValueError("Input must be a DataFrame or a list of ERB objects") 