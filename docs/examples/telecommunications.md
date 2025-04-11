# Telecommunications Network Analysis

This documentation provides an overview of the telecommunications network analysis capabilities in the Grapho Terrain package.

## Overview

The telecommunications module provides tools for analyzing cellular networks, particularly Estações Rádio Base (ERBs) or cellular base stations. It includes functionality for:

1. Processing ERB data
2. Computing radio signal coverage areas
3. Visualizing coverage and network properties
4. Analyzing network topology and metrics
5. Creating geospatial graphs from ERB data

## Key Components

### ERB Data Model

The `ERB` class represents a cellular base station with properties including:

- Location (geometry)
- Operator information
- Technical parameters (frequency, power, gain, etc.)
- Antenna properties (azimuth, height)

ERB objects are managed in an `ERBLayer` that provides collective operations on multiple base stations.

### Coverage Analysis

The coverage module includes functions for computing:

- EIRP (Effective Isotropic Radiated Power)
- Coverage radius based on technical parameters and environment
- Precise coverage sectors
- Line-of-sight calculations
- Terrain-aware coverage modeling
- Fresnel zone analysis

### Network Analysis

The network analysis module provides:

- ERB connectivity graph creation
- Voronoi diagram generation for coverage areas
- Overlap analysis
- Network metrics computation

## Example Usage

### Loading ERB Data

```python
from grapho_terrain.telecommunications.erb import ERBLayer, carregar_dados_erbs

# Load from CSV
erb_layer = carregar_dados_erbs("path/to/erbs.csv")

# Or create from GeoDataFrame
erb_layer = ERBLayer.from_geodataframe(gdf_erbs)
```

### Computing Coverage

```python
from grapho_terrain.telecommunications.coverage import calcular_eirp, calcular_raio_cobertura_aprimorado

# Calculate EIRP
potencia_watts = 40
ganho_antena = 17  # dBi
eirp = calcular_eirp(potencia_watts, ganho_antena)

# Calculate coverage radius
freq_mhz = 2600
raio = calcular_raio_cobertura_aprimorado(eirp, freq_mhz, tipo_area='urbana')

# Create coverage sectors for all ERBs
gdf_sectors = erb_layer.create_coverage_sectors(tipo_area='urbana')
```

### Visualizing ERB Data

```python
from grapho_terrain.visualization.urban import plot_erb_locations, plot_erb_coverage

# Plot ERB locations
fig, ax = plt.subplots(figsize=(10, 8))
plot_erb_locations(gdf_erbs, ax=ax, by_operator=True, add_basemap=True)

# Plot coverage areas
fig, ax = plt.subplots(figsize=(10, 8))
plot_erb_coverage(gdf_erbs, gdf_sectors, ax=ax, by_operator=True)
```

### Creating Network Graphs

```python
from grapho_terrain.telecommunications.network import create_erb_graph, analyze_network_metrics

# Create a graph connecting nearest neighbors
erb_graph = create_erb_graph(erb_layer, k_nearest=3)

# Analyze network properties
metrics = analyze_network_metrics(erb_graph)
```

## Integration with Terrain Analysis

The telecommunications module can be integrated with terrain data to perform more realistic coverage analysis:

```python
from grapho_terrain.core.data_model import TerrainModel
from grapho_terrain.telecommunications.coverage import calc_effective_radius

# Load terrain data
terrain_model = TerrainModel(name="Sample Terrain")
terrain_model.set_dem(dem_data, dem_transform)

# Calculate effective coverage considering terrain
erb = erb_layer.get_erb("ERB_1")
lat, lon = erb.geometry.y, erb.geometry.x
azimuth = erb.azimute
radius = erb.calculate_coverage_radius()

# Calculate effective coverage with terrain consideration
effective_coverage = calc_effective_radius(
    lat, lon, azimuth, terrain_model, radius,
    considerar_difracao=True, considerar_reflexao=True
)
```

## Integration with Multi-Layer Graphs

ERB data can be included as a layer in a multi-layer feature graph:

```python
from grapho_terrain.network.feature_graphs import MultiLayerFeatureGraph

# Create a multi-layer graph with ERBs, buildings, and roads
dataframes = {
    'erbs': gdf_erbs,
    'buildings': gdf_buildings,
    'roads': gdf_roads
}

# Define features for each layer
features = {
    'erbs': ['freq_mhz', 'potencia_watts', 'ganho_antena'],
    'buildings': ['height', 'floors'],
    'roads': ['lanes', 'speed_limit']
}

# Create the multi-layer graph
multi_graph = MultiLayerFeatureGraph.from_layer_dataframes(
    dataframes, features=features, k_intra=3, k_inter=2
)

# Export to PyTorch Geometric for ML
hetero_data = multi_graph.to_torch_geometric_hetero()
``` 