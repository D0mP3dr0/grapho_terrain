"""
Functions for visualizing urban data.

This module provides functions for visualizing urban features
such as buildings, roads, land use, and telecommunications infrastructure.
"""

import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

def plot_urban_features(gdf, ax=None, figsize=(10, 8), column=None, cmap='viridis',
                       markersize=30, legend=True, title=None, add_basemap=False):
    """
    Plot urban features from a GeoDataFrame.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing urban features
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    column : str, optional
        Column to use for coloring features
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use
    markersize : int, optional
        Size of markers for point features
    legend : bool, optional
        Whether to include a legend
    title : str, optional
        Plot title
    add_basemap : bool, optional
        Whether to add a basemap
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the features
    gdf.plot(ax=ax, column=column, cmap=cmap, markersize=markersize, legend=legend)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=gdf.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_building_heights(gdf_buildings, ax=None, figsize=(10, 8), height_col='height',
                        cmap='viridis', title=None, add_basemap=False):
    """
    Plot buildings colored by height.
    
    Parameters
    ----------
    gdf_buildings : GeoDataFrame
        GeoDataFrame containing building polygons
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    height_col : str, optional
        Column containing building height values
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use
    title : str, optional
        Plot title
    add_basemap : bool, optional
        Whether to add a basemap
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Check if height column exists
    if height_col not in gdf_buildings.columns:
        print(f"Warning: Column '{height_col}' not found. Using uniform color.")
        gdf_buildings.plot(ax=ax, color='lightgray')
    else:
        # Plot buildings colored by height
        gdf_buildings.plot(ax=ax, column=height_col, cmap=cmap, legend=True)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=gdf_buildings.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Building Heights")
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_erb_locations(gdf_erb, ax=None, figsize=(10, 8), by_operator=False, 
                     marker_size=30, add_basemap=False, title=None):
    """
    Plot cellular base station (ERB) locations.
    
    Parameters
    ----------
    gdf_erb : GeoDataFrame
        GeoDataFrame containing ERB points
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    by_operator : bool, optional
        Whether to color points by operator
    marker_size : int, optional
        Size of markers
    add_basemap : bool, optional
        Whether to add a basemap
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Check if the operadora column exists when coloring by operator
    if by_operator:
        # Look for operator column with different possible names
        operator_cols = ['operadora', 'OPERADORA', 'operator', 'empresa']
        operator_col = next((col for col in operator_cols if col in gdf_erb.columns), None)
        
        if operator_col:
            # Get unique operators
            operators = gdf_erb[operator_col].unique()
            
            # Create a colormap
            colors = plt.cm.tab10(np.linspace(0, 1, len(operators)))
            color_dict = dict(zip(operators, colors))
            
            # Plot each operator with a different color
            for operator, color in color_dict.items():
                subset = gdf_erb[gdf_erb[operator_col] == operator]
                subset.plot(ax=ax, color=color, markersize=marker_size, 
                          label=operator, alpha=0.7)
                
            # Add legend
            ax.legend(title="Operator")
        else:
            print("Warning: Operator column not found. Using uniform color.")
            gdf_erb.plot(ax=ax, color='red', markersize=marker_size)
    else:
        # Plot all ERBs with the same color
        gdf_erb.plot(ax=ax, color='red', markersize=marker_size)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=gdf_erb.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Cellular Base Station Locations")
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_erb_coverage(gdf_erb, gdf_sectors, ax=None, figsize=(10, 8), by_operator=True,
                    alpha=0.5, add_basemap=False, title=None):
    """
    Plot cellular coverage areas.
    
    Parameters
    ----------
    gdf_erb : GeoDataFrame
        GeoDataFrame containing ERB points
    gdf_sectors : GeoDataFrame
        GeoDataFrame containing coverage sector polygons
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    by_operator : bool, optional
        Whether to color coverage areas by operator
    alpha : float, optional
        Transparency of coverage areas
    add_basemap : bool, optional
        Whether to add a basemap
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Check if the ERB ID column exists in the sectors DataFrame
    erb_id_cols = ['erb_id', 'id_erb', 'ID_ERB']
    erb_id_col = next((col for col in erb_id_cols if col in gdf_sectors.columns), None)
    
    if not erb_id_col:
        print("Warning: ERB ID column not found in sectors data. Cannot relate to ERBs.")
        gdf_sectors.plot(ax=ax, color='blue', alpha=alpha)
        gdf_erb.plot(ax=ax, color='red', markersize=30)
    else:
        if by_operator:
            # Look for operator column with different possible names
            operator_cols = ['operadora', 'OPERADORA', 'operator', 'empresa']
            operator_col = next((col for col in operator_cols if col in gdf_sectors.columns), None)
            
            if operator_col:
                # Get unique operators
                operators = gdf_sectors[operator_col].unique()
                
                # Create a colormap
                colors = plt.cm.tab10(np.linspace(0, 1, len(operators)))
                color_dict = dict(zip(operators, colors))
                
                # Plot each operator's coverage with a different color
                for operator, color in color_dict.items():
                    subset = gdf_sectors[gdf_sectors[operator_col] == operator]
                    subset.plot(ax=ax, color=color, alpha=alpha, label=operator)
                    
                # Add legend
                ax.legend(title="Operator")
            else:
                print("Warning: Operator column not found. Using uniform color.")
                gdf_sectors.plot(ax=ax, color='blue', alpha=alpha)
        else:
            # Plot all coverage areas with the same color
            gdf_sectors.plot(ax=ax, color='blue', alpha=alpha)
    
    # Plot ERB locations
    gdf_erb.plot(ax=ax, color='red', markersize=20)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=gdf_erb.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Cellular Coverage Areas")
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_coverage_overlap(gdf_sectors, ax=None, figsize=(10, 8), add_basemap=False, title=None):
    """
    Plot cellular coverage overlap areas.
    
    Parameters
    ----------
    gdf_sectors : GeoDataFrame
        GeoDataFrame containing coverage sector polygons
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    add_basemap : bool, optional
        Whether to add a basemap
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create an overlay of all sectors to find overlaps
    import geopandas as gpd
    from shapely.ops import unary_union
    
    # Initialize count raster (we'll use this to count overlaps)
    bbox = gdf_sectors.total_bounds
    cell_size = 0.001  # Adjust based on your data
    
    # Create a grid of points
    x = np.arange(bbox[0], bbox[2], cell_size)
    y = np.arange(bbox[1], bbox[3], cell_size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize overlap count
    Z = np.zeros_like(X)
    
    # Count overlaps for each cell
    for i in range(len(X)):
        for j in range(len(X[0])):
            point = gpd.points_from_xy([X[i, j]], [Y[i, j]], crs=gdf_sectors.crs)[0]
            count = sum(gdf_sectors.geometry.contains(point))
            Z[i, j] = count
    
    # Plot overlap counts
    img = ax.imshow(Z, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), 
                  origin='lower', cmap='viridis', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Number of Overlapping Coverage Areas')
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=gdf_sectors.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Cellular Coverage Overlap")
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_erb_heatmap(gdf_erb, ax=None, figsize=(10, 8), radius=0.01,
                   attribute=None, add_basemap=False, title=None):
    """
    Create a heatmap of ERB density or other attributes.
    
    Parameters
    ----------
    gdf_erb : GeoDataFrame
        GeoDataFrame containing ERB points
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    radius : float, optional
        Radius for density calculation in decimal degrees
    attribute : str, optional
        Attribute to use for heatmap values (if None, uses density)
    add_basemap : bool, optional
        Whether to add a basemap
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get bounds of the data
    bounds = gdf_erb.total_bounds
    
    # Create a grid of points for the heatmap
    x_range = np.linspace(bounds[0], bounds[2], 100)
    y_range = np.linspace(bounds[1], bounds[3], 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Create value matrix
    Z = np.zeros_like(X)
    
    # Calculate the value for each grid cell
    for i in range(len(X)):
        for j in range(len(X[0])):
            from shapely.geometry import Point
            
            # Create a point at this location
            point = Point(X[i, j], Y[i, j])
            
            if attribute is None:
                # Calculate ERB density
                from ..telecommunications.erb import calcular_densidade_erb
                Z[i, j] = calcular_densidade_erb(point, gdf_erb, radius)
            else:
                # Use attribute values weighted by distance
                if attribute in gdf_erb.columns:
                    # Calculate distances to all ERBs
                    distances = []
                    values = []
                    for idx, row in gdf_erb.iterrows():
                        # Skip if attribute value is missing
                        if pd.isna(row[attribute]):
                            continue
                            
                        # Calculate distance
                        dx = (row.geometry.x - point.x) * 111.32 * np.cos(np.radians((row.geometry.y + point.y) / 2))
                        dy = (row.geometry.y - point.y) * 111.32
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        
                        if distance <= radius * 111.32:  # Convert radius to km
                            distances.append(distance)
                            values.append(row[attribute])
                    
                    # Calculate inverse distance weighted average
                    if len(distances) > 0:
                        weights = 1.0 / (np.array(distances) + 0.1)  # Add small constant to avoid division by zero
                        Z[i, j] = np.average(values, weights=weights)
                else:
                    print(f"Warning: Attribute '{attribute}' not found in ERB data.")
                    Z[i, j] = 0
    
    # Plot the heatmap
    img = ax.imshow(Z, extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
                  origin='lower', cmap='hot', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax)
    if attribute is None:
        cbar.set_label('ERB Density (count/km²)')
    else:
        cbar.set_label(attribute)
    
    # Plot ERB locations
    gdf_erb.plot(ax=ax, color='blue', markersize=10, alpha=0.5)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=gdf_erb.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        if attribute is None:
            ax.set_title("ERB Density Heatmap")
        else:
            ax.set_title(f"ERB {attribute} Heatmap")
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax 