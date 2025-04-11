"""
Functions for visualizing terrain data.

This module provides functions for visualizing terrain models,
including contour lines, digital elevation models, and 3D terrain.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D
import contextily as ctx
from ..core.data_model import TerrainModel


def plot_contours(contours_gdf, ax=None, figsize=(10, 8), cmap='viridis', 
                  column='elevation', legend=True, title='Contour Lines', add_basemap=False):
    """
    Plot contour lines from a GeoDataFrame.
    
    Parameters
    ----------
    contours_gdf : GeoDataFrame
        GeoDataFrame containing contour lines
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use
    column : str, optional
        Column to use for coloring
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
    
    # Plot contours
    contours_gdf.plot(ax=ax, column=column, cmap=cmap, legend=legend, 
                      linewidth=0.5, alpha=0.7)
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=contours_gdf.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Keep axes equal for proper geographic visualization
    ax.set_aspect('equal')
    
    return ax


def plot_dem(dem_data, transform=None, ax=None, figsize=(10, 8), cmap='terrain', 
             title='Digital Elevation Model', colorbar=True, hillshade=False, alpha=1.0):
    """
    Plot a digital elevation model as a raster.
    
    Parameters
    ----------
    dem_data : numpy.ndarray
        2D array containing elevation data
    transform : affine.Affine, optional
        Affine transform for the raster
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use
    title : str, optional
        Plot title
    colorbar : bool, optional
        Whether to include a colorbar
    hillshade : bool, optional
        Whether to apply hillshading
    alpha : float, optional
        Transparency of the raster
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create a hillshade if requested
    if hillshade:
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshaded = ls.hillshade(dem_data, vert_exag=0.3)
        ax.imshow(hillshaded, cmap='gray', alpha=0.5)
    
    # Plot the DEM
    img = ax.imshow(dem_data, cmap=cmap, alpha=alpha)
    
    # Add colorbar if requested
    if colorbar:
        cbar = plt.colorbar(img, ax=ax, shrink=0.5)
        cbar.set_label('Elevation')
    
    # Add title
    ax.set_title(title)
    
    # Add gridlines
    ax.grid(linestyle='--', alpha=0.3)
    
    return ax


def plot_terrain_model(terrain, figsize=(12, 10), add_basemap=False, 
                       dem_cmap='terrain', contour_cmap='viridis'):
    """
    Plot a terrain model, showing both DEM and contours in subplots.
    
    Parameters
    ----------
    terrain : TerrainModel
        Terrain model to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    add_basemap : bool, optional
        Whether to add a basemap to the contour plot
    dem_cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for the DEM
    contour_cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for contours
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot DEM if available
    if terrain.dem is not None:
        plot_dem(terrain.dem, transform=terrain.dem_transform, ax=axes[0],
                cmap=dem_cmap, title='Digital Elevation Model', hillshade=True)
    else:
        axes[0].text(0.5, 0.5, 'No DEM data available', ha='center', va='center')
        axes[0].set_title('Digital Elevation Model (No Data)')
    
    # Plot contours if available
    if terrain.contours is not None:
        # Identify the elevation column
        elev_cols = ['elevation', 'elev', 'height', 'z', 'alt', 'altitude', 'contour']
        elev_col = next((col for col in elev_cols if col in terrain.contours.columns), None)
        
        if elev_col:
            plot_contours(terrain.contours, ax=axes[1], cmap=contour_cmap, 
                         column=elev_col, title='Contour Lines', add_basemap=add_basemap)
        else:
            # If no elevation column is found, just plot the geometries
            terrain.contours.plot(ax=axes[1], linewidth=0.5, color='blue')
            axes[1].set_title('Contour Lines (No Elevation Data)')
    else:
        axes[1].text(0.5, 0.5, 'No contour data available', ha='center', va='center')
        axes[1].set_title('Contour Lines (No Data)')
    
    # Add main title
    if terrain.name:
        fig.suptitle(f'Terrain Model: {terrain.name}', fontsize=16)
    else:
        fig.suptitle('Terrain Model', fontsize=16)
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_terrain_3d(terrain, figsize=(12, 10), elev=45, azim=225, cmap='terrain', 
                     alpha=0.8, wireframe=False, vertical_exaggeration=2):
    """
    Create a 3D visualization of a terrain model.
    
    Parameters
    ----------
    terrain : TerrainModel
        Terrain model to visualize
    figsize : tuple, optional
        Figure size (width, height) in inches
    elev : float, optional
        Elevation angle for the 3D view
    azim : float, optional
        Azimuth angle for the 3D view
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use
    alpha : float, optional
        Transparency of the surface
    wireframe : bool, optional
        Whether to plot as a wireframe instead of a surface
    vertical_exaggeration : float, optional
        Vertical exaggeration factor
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 3D plot
    """
    # Check if there is valid data
    if terrain.dem is None:
        raise ValueError("Terrain model must have DEM data for 3D visualization")
    
    # Create a new figure with a 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create X and Y coordinate matrices
    ny, nx = terrain.dem.shape
    if terrain.dem_transform:
        x0, y0 = terrain.dem_transform * (0, 0)
        x1, y1 = terrain.dem_transform * (nx, ny)
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
    else:
        x = np.arange(0, nx)
        y = np.arange(0, ny)
    
    X, Y = np.meshgrid(x, y)
    
    # Apply vertical exaggeration
    Z = terrain.dem * vertical_exaggeration
    
    # Plot the surface or wireframe
    if wireframe:
        surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='black')
    else:
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, 
                              antialiased=True, alpha=alpha)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')
    
    # Set view angles
    ax.view_init(elev=elev, azim=azim)
    
    # Add labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel(f'Elevation (x{vertical_exaggeration})')
    
    # Add title
    if terrain.name:
        ax.set_title(f'3D Terrain Model: {terrain.name}')
    else:
        ax.set_title('3D Terrain Model')
    
    return fig


def plot_terrain_profile(terrain, start_point, end_point, num_points=100, 
                           figsize=(10, 6), ax=None, title=None):
    """
    Plot an elevation profile along a line between two points.
    
    Parameters
    ----------
    terrain : TerrainModel
        Terrain model to use
    start_point : tuple
        Start point (x, y)
    end_point : tuple
        End point (x, y)
    num_points : int, optional
        Number of points to sample along the line
    figsize : tuple, optional
        Figure size (width, height) in inches
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the profile plot
    """
    # Check if there is valid data
    if terrain.dem is None and terrain.contours is None:
        raise ValueError("Terrain model must have DEM or contour data for profile plotting")
    
    # Create a new axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Generate points along the line
    t = np.linspace(0, 1, num_points)
    x = start_point[0] + (end_point[0] - start_point[0]) * t
    y = start_point[1] + (end_point[1] - start_point[1]) * t
    
    # Calculate distance along the profile
    distance = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
    
    # Get elevation for each point
    elevations = []
    for i in range(num_points):
        elev = terrain.get_elevation(x[i], y[i])
        elevations.append(elev)
    
    # Convert to numpy array
    elevations = np.array(elevations)
    
    # Plot the profile
    ax.plot(distance, elevations, 'b-')
    ax.fill_between(distance, 0, elevations, alpha=0.3, color='blue')
    
    # Add labels
    ax.set_xlabel('Distance')
    ax.set_ylabel('Elevation')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Terrain Profile')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax 