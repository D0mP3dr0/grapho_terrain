# Grapho Terrain

A Python package for geospatial graph analysis of terrain and urban features.

## Overview

Grapho Terrain is a research tool for creating, analyzing, and visualizing heterogeneous geospatial graphs that integrate terrain, climate, and urban infrastructure data. It provides a robust framework for researchers in urban planning, environmental science, and geospatial analysis.

## Features

- **Topographic Processing**: Generate and analyze terrain models from contour data
- **Urban Feature Extraction**: Process and standardize buildings, roads, and other urban infrastructure
- **Climate Integration**: Add climate data layers to spatial models
- **Graph Construction**: Create multi-layered graphs from heterogeneous geospatial data
- **3D Visualization**: Advanced visualization tools for terrain, urban features, and analytical results
- **Population Analysis**: Tools for modeling population distribution across space and time
- **Integrated Pipeline**: End-to-end workflow from data loading to graph analysis and visualization
- **Telecommunications Analysis**: ERB (Radio Base Station) coverage analysis and network modeling

## Installation

```bash
# Clone the repository
git clone https://github.com/ufabc/grapho_terrain.git
cd grapho_terrain

# Install the package and dependencies
pip install -e .
```

## Usage

### Basic Usage

```python
import grapho_terrain as gt

# Load terrain data
contours = gt.io.load_topography("path/to/contours.gpkg")

# Create a DEM from contours
dem = gt.processing.create_dem_from_contours(contours, resolution=30)

# Visualize the terrain
gt.visualization.visualize_terrain_3d(dem, title="3D Terrain Model")
```

### Using the Complete Pipeline

For a complete end-to-end workflow, use one of our pipeline scripts:

```bash
# Run the simplified pipeline (recommended for first-time users)
python examples/pipeline_simples.py

# Generate synthetic test data automatically
python examples/pipeline_simples.py

# Use the comprehensive pipeline (requires all dependencies)
python examples/pipeline_completo.py --config examples/pipeline_config.json

# Use the Sorocaba ERB example
python examples/pipeline_completo.py --sorocaba
```

See `examples/README_pipeline.md` for complete documentation on the pipeline.

## Project Structure

```
grapho_terrain/
├── grapho_terrain/        # Main package
│   ├── core/              # Core functionality
│   ├── io/                # Input/output operations
│   ├── processing/        # Data processing modules
│   ├── visualization/     # Visualization tools
│   ├── network/           # Graph/network analysis
│   ├── telecommunications/ # ERB and telecom analysis
│   ├── pipeline/          # Integrated workflow components
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── docs/                  # Documentation
├── examples/              # Example usage scripts
│   ├── pipeline_completo.py  # Complete integrated pipeline
│   ├── pipeline_simples.py   # Simplified pipeline
│   ├── sorocaba/          # Sorocaba ERB examples
│   └── README_pipeline.md # Pipeline documentation
├── output/                # Generated outputs
└── tests/                 # Test suite
```

## Documentation

Comprehensive documentation is available in the `docs` directory and online at [https://grapho-terrain.readthedocs.io](https://grapho-terrain.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Grapho Terrain in your research, please cite:

```
@software{grapho_terrain,
  author = {UFABC},
  title = {Grapho Terrain: A Python Package for Geospatial Graph Analysis},
  year = {2023},
  url = {https://github.com/ufabc/grapho_terrain}
}
```

## Acknowledgements

This project was developed at the Federal University of ABC (UFABC) with support from [funding agencies/collaborators]. 