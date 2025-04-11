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