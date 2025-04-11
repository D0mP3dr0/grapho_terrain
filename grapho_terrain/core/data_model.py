"""
Core data models for geospatial data representation.

This module defines the standard data structures used throughout
the package for representing different types of geospatial data.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from typing import Dict, List, Tuple, Optional, Union, Any


class TerrainModel:
    """
    Representation of terrain data, including elevation models and contours.
    """
    
    def __init__(self, name=None, crs=None):
        """
        Initialize a TerrainModel.
        
        Parameters
        ----------
        name : str, optional
            Name of the terrain model
        crs : str or dict, optional
            Coordinate reference system of the terrain model
        """
        self.name = name
        self.crs = crs
        self.dem = None  # Digital Elevation Model
        self.dem_transform = None  # Affine transform for the DEM
        self.contours = None  # Contour lines as GeoDataFrame
        self.extent = None  # Geographic extent as tuple (xmin, ymin, xmax, ymax)
        self.resolution = None  # Resolution in units of the CRS
        self.metadata = {}  # Additional metadata
        
    def set_dem(self, dem_array, transform, resolution=None):
        """
        Set the Digital Elevation Model.
        
        Parameters
        ----------
        dem_array : numpy.ndarray
            2D array containing elevation values
        transform : affine.Affine
            Affine transform for converting array indices to coordinates
        resolution : float, optional
            Spatial resolution of the DEM
        """
        self.dem = dem_array
        self.dem_transform = transform
        if resolution:
            self.resolution = resolution
            
    def set_contours(self, contours_gdf):
        """
        Set contour lines for the terrain.
        
        Parameters
        ----------
        contours_gdf : GeoDataFrame
            GeoDataFrame containing contour lines with elevation values
        """
        self.contours = contours_gdf
        
        # Update CRS if not already set
        if not self.crs and contours_gdf.crs:
            self.crs = contours_gdf.crs
            
        # Update extent from contours
        if contours_gdf.shape[0] > 0:
            self.extent = tuple(contours_gdf.total_bounds)
            
    def set_extent(self, extent):
        """
        Set the geographic extent of the terrain model.
        
        Parameters
        ----------
        extent : tuple
            Geographic extent as (xmin, ymin, xmax, ymax)
        """
        self.extent = extent
        
    def get_elevation(self, x, y):
        """
        Get the elevation at a specific point.
        
        Parameters
        ----------
        x : float
            X-coordinate
        y : float
            Y-coordinate
            
        Returns
        -------
        float or None
            Elevation value at the point, or None if outside the extent
        """
        if self.dem is not None and self.dem_transform is not None:
            # Check if point is within extent
            if self.extent:
                xmin, ymin, xmax, ymax = self.extent
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    return None
                    
            # Convert coordinates to array indices
            col, row = ~self.dem_transform * (x, y)
            col, row = int(col), int(row)
            
            # Check if indices are within array bounds
            if 0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]:
                return self.dem[row, col]
                
        return None
        

class UrbanFeature:
    """
    Base class for urban features (buildings, roads, etc.).
    """
    
    def __init__(self, feature_id, geometry, feature_type, attributes=None, crs=None):
        """
        Initialize an UrbanFeature.
        
        Parameters
        ----------
        feature_id : str or int
            Unique identifier for the feature
        geometry : shapely.geometry
            Geometry of the feature
        feature_type : str
            Type of urban feature (e.g., 'building', 'road', 'park')
        attributes : dict, optional
            Dictionary of feature attributes
        crs : str or dict, optional
            Coordinate reference system
        """
        self.id = feature_id
        self.geometry = geometry
        self.feature_type = feature_type
        self.attributes = attributes or {}
        self.crs = crs
        
    def __repr__(self):
        return f"UrbanFeature(id={self.id}, type={self.feature_type})"
    
    def to_geoseries(self):
        """
        Convert the feature to a GeoSeries.
        
        Returns
        -------
        GeoSeries
            GeoSeries representation of the feature
        """
        data = {"id": self.id, "feature_type": self.feature_type, **self.attributes}
        return gpd.GeoSeries(data=data, geometry=self.geometry, crs=self.crs)


class Building(UrbanFeature):
    """
    Representation of a building feature.
    """
    
    def __init__(self, building_id, geometry, height=None, floors=None, 
                 building_type=None, attributes=None, crs=None):
        """
        Initialize a Building.
        
        Parameters
        ----------
        building_id : str or int
            Unique identifier for the building
        geometry : shapely.geometry.Polygon
            Geometry of the building
        height : float, optional
            Height of the building in meters
        floors : int, optional
            Number of floors in the building
        building_type : str, optional
            Type of building (e.g., 'residential', 'commercial')
        attributes : dict, optional
            Dictionary of building attributes
        crs : str or dict, optional
            Coordinate reference system
        """
        # Initialize the base class
        super().__init__(building_id, geometry, "building", attributes, crs)
        
        # Add building-specific attributes
        self.height = height
        self.floors = floors
        self.building_type = building_type
        
        # Add these to attributes dictionary as well
        if height is not None:
            self.attributes["height"] = height
        if floors is not None:
            self.attributes["floors"] = floors
        if building_type is not None:
            self.attributes["building_type"] = building_type


class Road(UrbanFeature):
    """
    Representation of a road feature.
    """
    
    def __init__(self, road_id, geometry, road_type=None, name=None, 
                 width=None, attributes=None, crs=None):
        """
        Initialize a Road.
        
        Parameters
        ----------
        road_id : str or int
            Unique identifier for the road
        geometry : shapely.geometry.LineString
            Geometry of the road
        road_type : str, optional
            Type of road (e.g., 'primary', 'residential')
        name : str, optional
            Name of the road
        width : float, optional
            Width of the road in meters
        attributes : dict, optional
            Dictionary of road attributes
        crs : str or dict, optional
            Coordinate reference system
        """
        # Initialize the base class
        super().__init__(road_id, geometry, "road", attributes, crs)
        
        # Add road-specific attributes
        self.road_type = road_type
        self.name = name
        self.width = width
        
        # Add these to attributes dictionary as well
        if road_type is not None:
            self.attributes["road_type"] = road_type
        if name is not None:
            self.attributes["name"] = name
        if width is not None:
            self.attributes["width"] = width


class LandCover(UrbanFeature):
    """
    Representation of a land cover feature.
    """
    
    def __init__(self, landcover_id, geometry, landcover_class=None, 
                 attributes=None, crs=None):
        """
        Initialize a LandCover feature.
        
        Parameters
        ----------
        landcover_id : str or int
            Unique identifier for the land cover feature
        geometry : shapely.geometry.Polygon
            Geometry of the land cover feature
        landcover_class : str, optional
            Classification of land cover (e.g., 'forest', 'water', 'urban')
        attributes : dict, optional
            Dictionary of land cover attributes
        crs : str or dict, optional
            Coordinate reference system
        """
        # Initialize the base class
        super().__init__(landcover_id, geometry, "landcover", attributes, crs)
        
        # Add landcover-specific attributes
        self.landcover_class = landcover_class
        
        # Add to attributes dictionary as well
        if landcover_class is not None:
            self.attributes["landcover_class"] = landcover_class


class UrbanLayer:
    """
    A collection of urban features of a specific type.
    """
    
    def __init__(self, name, feature_type, crs=None):
        """
        Initialize an UrbanLayer.
        
        Parameters
        ----------
        name : str
            Name of the layer
        feature_type : str
            Type of features in the layer (e.g., 'building', 'road')
        crs : str or dict, optional
            Coordinate reference system
        """
        self.name = name
        self.feature_type = feature_type
        self.crs = crs
        self.features = {}
        self.metadata = {}
        
    def add_feature(self, feature):
        """
        Add a feature to the layer.
        
        Parameters
        ----------
        feature : UrbanFeature
            Feature to add
        """
        if feature.feature_type != self.feature_type:
            raise ValueError(f"Feature type '{feature.feature_type}' does not match layer type '{self.feature_type}'")
            
        self.features[feature.id] = feature
        
        # Update CRS if not already set
        if not self.crs and feature.crs:
            self.crs = feature.crs
            
    def get_feature(self, feature_id):
        """
        Get a feature by ID.
        
        Parameters
        ----------
        feature_id : str or int
            ID of the feature
            
        Returns
        -------
        UrbanFeature
            The feature with the given ID
        """
        return self.features.get(feature_id)
        
    def to_geodataframe(self):
        """
        Convert the layer to a GeoDataFrame.
        
        Returns
        -------
        GeoDataFrame
            GeoDataFrame containing all features in the layer
        """
        features_data = []
        for feature_id, feature in self.features.items():
            data = {
                "id": feature_id,
                "feature_type": feature.feature_type,
                "geometry": feature.geometry
            }
            data.update(feature.attributes)
            features_data.append(data)
            
        gdf = gpd.GeoDataFrame(features_data, crs=self.crs)
        return gdf
        
    @classmethod
    def from_geodataframe(cls, gdf, name, feature_type, id_col="id"):
        """
        Create an UrbanLayer from a GeoDataFrame.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame containing the features
        name : str
            Name of the layer
        feature_type : str
            Type of features in the layer
        id_col : str, optional
            Name of the column containing feature IDs
            
        Returns
        -------
        UrbanLayer
            A new UrbanLayer instance
        """
        layer = cls(name, feature_type, crs=gdf.crs)
        
        for _, row in gdf.iterrows():
            feature_id = row.get(id_col, str(len(layer.features)))
            geometry = row.geometry
            
            # Extract attributes (exclude id and geometry)
            attributes = row.drop([id_col, "geometry"]).to_dict()
            
            # Create appropriate feature based on type
            if feature_type == "building":
                feature = Building(feature_id, geometry, 
                                   height=attributes.get("height"),
                                   floors=attributes.get("floors"),
                                   building_type=attributes.get("building_type"),
                                   attributes=attributes, crs=gdf.crs)
            elif feature_type == "road":
                feature = Road(feature_id, geometry,
                              road_type=attributes.get("road_type"),
                              name=attributes.get("name"),
                              width=attributes.get("width"),
                              attributes=attributes, crs=gdf.crs)
            elif feature_type == "landcover":
                feature = LandCover(feature_id, geometry,
                                   landcover_class=attributes.get("landcover_class"),
                                   attributes=attributes, crs=gdf.crs)
            else:
                feature = UrbanFeature(feature_id, geometry, feature_type,
                                     attributes=attributes, crs=gdf.crs)
                
            layer.add_feature(feature)
            
        return layer 