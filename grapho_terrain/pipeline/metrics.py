"""
Métricas avançadas para análise de grafos geoespaciais.

Este módulo implementa métricas avançadas para análise de grafos,
incluindo métricas espaciais específicas para redes geográficas
e métricas para grafos multicamada.
"""

import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from ..network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
import warnings # Added

# Optional RAPIDS imports
try:
    import cugraph
    import cudf
    RAPIDS_AVAILABLE = True
    warnings.warn("RAPIDS (cuGraph, cuDF) found. GPU acceleration will be available for graph algorithms.")
except ImportError:
    RAPIDS_AVAILABLE = False
    warnings.warn("RAPIDS (cuGraph, cuDF) not found. GPU acceleration will NOT be available for graph algorithms.")

# Check for CuPy availability (for matrix operations)
try:
    import cupy as cp
    # Test basic functionality to make sure CuPy works
    try:
        # Try to create a simple array and move it to the GPU
        test_array = cp.array([1, 2, 3])
        # Try a simple operation
        test_result = cp.sum(test_array)
        CUPY_AVAILABLE = True
        warnings.warn(f"CuPy found and working properly. GPU acceleration will be available for matrix operations. CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    except Exception as e:
        warnings.warn(f"CuPy found but failed to execute basic operations: {e}. Falling back to CPU.")
        CUPY_AVAILABLE = False
        cp = np  # Fall back to numpy if cupy is not available
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fall back to numpy if cupy is not available
    warnings.warn("CuPy not found. GPU acceleration will NOT be available for matrix operations.")

# Check for Numba availability (alternative GPU acceleration)
try:
    import numba
    from numba import cuda
    # Test if CUDA is available in Numba
    if cuda.is_available():
        NUMBA_CUDA_AVAILABLE = True
        warnings.warn(f"Numba CUDA found and working properly. GPU acceleration will be available for specific operations. CUDA capability: {cuda.get_current_device().compute_capability}")
    else:
        NUMBA_CUDA_AVAILABLE = False
        warnings.warn("Numba found but CUDA is not available. Will use CPU implementation.")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    warnings.warn("Numba not found. Cannot use GPU acceleration for specific operations.")

# Define Numba accelerated functions if available
if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def _gini_coefficient_kernel(data, result):
        """CUDA kernel for computing Gini coefficient"""
        # This is just one example of a simple operation that can be accelerated
        # For production use, this would be expanded to handle more complex operations
        n = data.shape[0]
        if n <= 1:
            result[0] = 0.0
            return
        
        # Perform the summation part of Gini calculation on GPU
        i = cuda.grid(1)
        if i < n:
            for j in range(n):
                cuda.atomic.add(result, 0, abs(data[i] - data[j]))
                
    def compute_gini_gpu(values):
        """Compute Gini coefficient using CUDA"""
        if len(values) <= 1:
            return 0.0
            
        # Move data to GPU
        data_gpu = cuda.to_device(np.array(values, dtype=np.float32))
        result_gpu = cuda.to_device(np.zeros(1, dtype=np.float32))
        
        # Configure grid
        threads_per_block = 256
        blocks_per_grid = (len(values) + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        _gini_coefficient_kernel[blocks_per_grid, threads_per_block](data_gpu, result_gpu)
        
        # Get result back from GPU
        result = result_gpu.copy_to_host()[0]
        
        # Complete Gini calculation
        n = len(values)
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return result / (2 * n * n * mean)
        
    @cuda.jit
    def _trapz_kernel(x, y, result):
        """CUDA kernel for computing trapezoidal integration (area under curve)"""
        n = x.shape[0]
        if n <= 1:
            result[0] = 0.0
            return
            
        # Each thread will handle a portion of the summation
        i = cuda.grid(1)
        if i < n - 1:
            area = (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2.0
            cuda.atomic.add(result, 0, area)
            
    def compute_trapz_gpu(x, y):
        """Compute trapezoidal integration using CUDA"""
        if len(x) <= 1 or len(y) <= 1:
            return 0.0
            
        # Ensure arrays are the same length
        n = min(len(x), len(y))
        
        # Move data to GPU
        x_gpu = cuda.to_device(np.array(x[:n], dtype=np.float32))
        y_gpu = cuda.to_device(np.array(y[:n], dtype=np.float32))
        result_gpu = cuda.to_device(np.zeros(1, dtype=np.float32))
        
        # Configure grid
        threads_per_block = 256
        blocks_per_grid = (n - 1 + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        _trapz_kernel[blocks_per_grid, threads_per_block](x_gpu, y_gpu, result_gpu)
        
        # Get result back from GPU
        return result_gpu.copy_to_host()[0]

    @cuda.jit
    def _matrix_product_kernel(matrixA, matrixB, result):
        """CUDA kernel for computing matrix product (for spatial autocorrelation)"""
        i, j = cuda.grid(2)
        if i < matrixA.shape[0] and j < matrixB.shape[1]:
            tmp = 0.0
            for k in range(matrixA.shape[1]):
                tmp += matrixA[i, k] * matrixB[k, j]
            result[i, j] = tmp
            
    def compute_matrix_product_gpu(A, B):
        """Compute matrix product using CUDA"""
        # Ensure inputs are 2D numpy arrays
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        
        # Check shapes
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix shapes incompatible for multiplication")
            
        # Allocate output array
        result_shape = (A.shape[0], B.shape[1])
        result = np.zeros(result_shape, dtype=np.float32)
        
        # Move to GPU
        A_gpu = cuda.to_device(A)
        B_gpu = cuda.to_device(B)
        result_gpu = cuda.to_device(result)
        
        # Configure grid
        threads_per_block = (16, 16)
        blocks_per_grid_x = (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (B.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Launch kernel
        _matrix_product_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, result_gpu)
        
        # Get result from GPU
        result_gpu.copy_to_host(result)
        return result
else:
    # Fallback to CPU implementation
    def compute_gini_gpu(values):
        """CPU fallback for Gini computation"""
        if len(values) <= 1:
            return 0.0
            
        # Sort values
        values_sorted = np.sort(values)
        n = len(values_sorted)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values_sorted) / (n * np.sum(values_sorted))) - (n + 1) / n
        
    def compute_trapz_gpu(x, y):
        """CPU fallback for trapezoidal integration"""
        return np.trapz(y, x=x)
        
    def compute_matrix_product_gpu(A, B):
        """CPU fallback for matrix product"""
        return np.dot(A, B)


class GraphMetrics:
    """
    Classe para calcular métricas avançadas de grafos.
    
    Esta classe fornece métodos para calcular métricas de grafos,
    incluindo métricas básicas, centralidade, comunidades e resiliência.
    """
    
    def __init__(self, graph, use_gpu=False):
        """
        Inicializa o calculador de métricas.
        
        Parameters
        ----------
        graph : networkx.Graph, FeatureGeoGraph
            Grafo a ser analisado
        use_gpu : bool, optional
            Se True, tenta usar cuGraph para aceleração em GPU (default: False)
        """
        if isinstance(graph, FeatureGeoGraph):
            self._feature_geograph = graph
            self.graph = graph.to_networkx()
        else:
            self._feature_geograph = None
            self.graph = graph
            
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()
        
        # Para algoritmos de grafos (RAPIDS)
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        
        # Para operações matriciais (CuPy)
        self.use_cupy = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_cupy else np
        
        # Pre-convert to cugraph if using GPU (can save time if called multiple times)
        self._cugraph_graph = None
        if self.use_gpu:
            try:
                # Convert networkx graph to cudf DataFrame for cugraph
                # Assuming unweighted for now, adjust if weights needed for basic metrics
                gdf = nx.to_pandas_edgelist(self.graph)
                gdf = cudf.from_pandas(gdf)

                # Create a cugraph Graph object
                self._cugraph_graph = cugraph.Graph()
                # Assuming graph is unweighted here. Need to adjust if weights are used by default
                self._cugraph_graph.from_cudf_edgelist(gdf, source='source', destination='target', renumber=True)
                warnings.warn("Successfully created cuGraph graph for GPU processing.")

            except Exception as e:
                warnings.warn(f"Failed to create cuGraph graph: {e}. Falling back to CPU.")
                self.use_gpu = False # Disable GPU use if conversion fails
                self._cugraph_graph = None

    def _get_cugraph_graph(self, weight_column=None):
        """Helper to get or create a cugraph.Graph, potentially weighted."""
        if not self.use_gpu:
            return None

        # If a weighted graph is needed and the pre-converted one is unweighted (or vice versa)
        # or if it wasn't created successfully initially.
        # This part needs careful handling based on how weights are used.
        # For simplicity, let's assume the pre-converted one is sufficient if it exists.
        # A more robust implementation would recreate it if weights change.
        if self._cugraph_graph:
             if weight_column and 'weights' not in self._cugraph_graph.edgelist.columns:
                 warnings.warn(f"Pre-computed cuGraph graph is unweighted, but weighted graph requested for {weight_column}. Attempting re-creation (may be slow).")
                 # Fall-through to recreate
             elif not weight_column and 'weights' in self._cugraph_graph.edgelist.columns:
                  warnings.warn(f"Pre-computed cuGraph graph is weighted, but unweighted graph requested. Attempting re-creation (may be slow).")
                  # Fall-through to recreate
             else:
                # Use pre-computed graph
                 return self._cugraph_graph

        # If no pre-computed graph or needs recreation
        try:
            gdf = nx.to_pandas_edgelist(self.graph)
            if weight_column and weight_column in gdf.columns:
                 # Ensure weight column is float type
                 gdf[weight_column] = gdf[weight_column].astype(np.float32)
                 edge_attr = [weight_column] # Pass weight column name for cugraph
            else:
                 weight_column = None # Ensure it's None if not found or not requested
                 edge_attr = None

            gdf = cudf.from_pandas(gdf)
            G_cu = cugraph.Graph()
            G_cu.from_cudf_edgelist(gdf, source='source', destination='target', edge_attr=edge_attr, renumber=True)
            warnings.warn(f"Created {'weighted' if weight_column else 'unweighted'} cuGraph graph on demand.")
            return G_cu
        except Exception as e:
            warnings.warn(f"Failed to create cuGraph graph on demand (weighted={weight_column}): {e}. Falling back to CPU.")
            self.use_gpu = False # Disable GPU for subsequent calls in this instance if on-demand fails
            return None
        
    def basic_metrics(self):
        """
        Calcula métricas básicas do grafo.
        
        Returns
        -------
        dict
            Dicionário com as métricas básicas
        """
        g = self.graph
        metrics = {
            'num_nodes': self.n,
            'num_edges': self.m,
            'density': nx.density(g),
            'avg_degree': 2 * self.m / self.n if self.n > 0 else 0
        }
        
        # Clustering coefficient (ignore self-loops if present)
        try:
            if self.use_gpu:
                 G_cu = self._get_cugraph_graph()
                 if G_cu:
                     # cuGraph returns clustering per node, need to average
                     df_cc = cugraph.clustering(G_cu)
                     metrics['avg_clustering'] = df_cc['clustering_coeff'].mean()
                     # Convert back to python float if it's a cudf scalar
                     if hasattr(metrics['avg_clustering'], 'item'):
                         metrics['avg_clustering'] = metrics['avg_clustering'].item()
                 else: # Fallback if G_cu failed
                      metrics['avg_clustering'] = nx.average_clustering(g)
            else:
                metrics['avg_clustering'] = nx.average_clustering(g)
        except Exception as e:
            warnings.warn(f"Could not calculate average clustering: {e}")
            metrics['avg_clustering'] = None
            
        # Add diameter and average path length for connected graphs
        if nx.is_connected(g):
            metrics['diameter'] = nx.diameter(g)
            metrics['avg_path_length'] = nx.average_shortest_path_length(g)
        else:
            # For disconnected graphs, compute metrics on the largest component
            largest_cc = max(nx.connected_components(g), key=len)
            largest_subgraph = g.subgraph(largest_cc).copy()
            
            if len(largest_cc) > 1:
                metrics['diameter_largest_cc'] = nx.diameter(largest_subgraph)
                metrics['avg_path_length_largest_cc'] = nx.average_shortest_path_length(largest_subgraph)
            else:
                metrics['diameter_largest_cc'] = 0
                metrics['avg_path_length_largest_cc'] = 0
                
            metrics['diameter'] = metrics['diameter_largest_cc']
            metrics['avg_path_length'] = metrics['avg_path_length_largest_cc']
            
        # Connected components
        metrics['connected_components'] = nx.number_connected_components(g)
        metrics['largest_cc_size'] = len(max(nx.connected_components(g), key=len))
        metrics['largest_cc_fraction'] = metrics['largest_cc_size'] / self.n if self.n > 0 else 0
        
        return metrics
        
    def centrality_metrics(self, normalized=True, include_nodes=False, k_betweenness=None, betweenness_subset_size=None):
        """
        Calcula métricas de centralidade para o grafo.
        
        Parameters
        ----------
        normalized : bool, optional
            Indica se as centralidades devem ser normalizadas
        include_nodes : bool, optional
            Indica se deve retornar os valores por nó ou apenas agregações
        k_betweenness : int, optional
            For GPU Betweenness: Number of source nodes to use for approximation.
            If None, uses exact calculation (can be slow/memory intensive).
        betweenness_subset_size : int, optional
             For GPU Betweenness: Size of the random subset of nodes to consider at each step.
             Used with k_betweenness.
            
        Returns
        -------
        dict
            Dicionário com as métricas de centralidade
        """
        metrics = {}
        centrality_values = {}
        
        # --- Define functions (GPU vs CPU) ---
        centrality_funcs_nx = {
            'degree': nx.degree_centrality,
            'betweenness': nx.betweenness_centrality,
            'closeness': nx.closeness_centrality
        }
        centrality_funcs_cugraph = {
            'degree': cugraph.degree_centrality,
            'betweenness': cugraph.betweenness_centrality,
            'closeness': cugraph.closeness_centrality # Note: cuGraph closeness might have different definition/normalization
        }
        
        # Eigenvector (Optional)
        try:
            import scipy.sparse # Required by nx eigenvector
            centrality_funcs_nx['eigenvector'] = nx.eigenvector_centrality_numpy # Use numpy version for better stability
            if RAPIDS_AVAILABLE:
                 centrality_funcs_cugraph['eigenvector'] = cugraph.eigenvector_centrality
        except ImportError:
            warnings.warn("SciPy not found, skipping NetworkX Eigenvector Centrality.")
        except Exception as e:
             warnings.warn(f"Cannot add Eigenvector Centrality: {e}")
        
        # PageRank (Optional)
        try:
            centrality_funcs_nx['pagerank'] = nx.pagerank
            if RAPIDS_AVAILABLE:
                 centrality_funcs_cugraph['pagerank'] = cugraph.pagerank
        except Exception as e:
            warnings.warn(f"Cannot add PageRank Centrality: {e}")
        
        # --- Select execution path ---
        use_gpu_for_centrality = self.use_gpu
        G_cu = None
        if use_gpu_for_centrality:
             G_cu = self._get_cugraph_graph() # Get unweighted graph for most centralities
             if G_cu is None:
                 use_gpu_for_centrality = False # Fallback if graph creation failed
        
        # --- Calculate Centralities ---
        if use_gpu_for_centrality:
            # --- GPU Path (cuGraph) ---
            warnings.warn("Using GPU (cuGraph) for centrality calculations.")
            for name, func in centrality_funcs_cugraph.items():
                try:
                    if name == 'betweenness':
                        # Use approximation if k is specified
                        result_df = func(G_cu, k=k_betweenness, normalized=normalized, seed=42, subset_size=betweenness_subset_size) # Pass k and subset_size
                    elif name == 'closeness':
                         # cuGraph closeness needs wf_normalized=True to match nx's normalization idea
                         result_df = func(G_cu, wf_normalized=True) # wf_normalized=True is default and recommended
                    else:
                         result_df = func(G_cu)

                    # Map back from renumbered vertices if needed
                    if G_cu.renumbered:
                         result_df = G_cu.unrenumber(result_df, 'vertex')

                    # Extract values and store
                    # Column names vary slightly in cuGraph results
                    value_col = name + '_centrality' if name + '_centrality' in result_df.columns else name # e.g., 'degree_centrality', 'pagerank'
                    if value_col not in result_df.columns:
                         # Handle cases like clustering where col name is different
                         if name == 'clustering' and 'clustering_coeff' in result_df.columns:
                             value_col = 'clustering_coeff'
                         else:
                             raise ValueError(f"Could not find result column for {name} in cuGraph output")

                    # Convert cudf Series to numpy array for stats
                    values_array = result_df[value_col].to_numpy()
                    nodes_order = result_df['vertex'].to_numpy() # Keep track of node order

                    # Store node-level results if requested
                    if include_nodes:
                         centrality_values[name] = dict(zip(nodes_order, values_array))

                    # Calculate aggregate statistics using numpy
                    metrics[f'{name}_mean'] = np.mean(values_array)
                    metrics[f'{name}_median'] = np.median(values_array)
                    metrics[f'{name}_std'] = np.std(values_array)
                    metrics[f'{name}_max'] = np.max(values_array)
                    metrics[f'{name}_min'] = np.min(values_array)

                    # Calculate Gini coefficient using Numba GPU acceleration if available
                    if NUMBA_CUDA_AVAILABLE:
                        warnings.warn(f"Using Numba CUDA for Gini coefficient calculation of {name} centrality.")
                        metrics[f'{name}_gini'] = compute_gini_gpu(values_array)
                    else:
                        # Traditional method
                        values_array_sorted = np.sort(values_array)
                        n_vals = len(values_array_sorted)
                        if n_vals > 0 and np.sum(values_array_sorted) != 0:
                            index = np.arange(1, n_vals + 1)
                            metrics[f'{name}_gini'] = (2 * np.sum(index * values_array_sorted) / (n_vals * np.sum(values_array_sorted))) - (n_vals + 1) / n_vals
                        else:
                            metrics[f'{name}_gini'] = 0 if n_vals > 0 else None # Gini is 0 if all values are same, None if empty

                    # Calculate deciles
                    deciles = [np.percentile(values_array, p) for p in range(0, 101, 10)]
                    for i, p in enumerate(range(0, 101, 10)):
                        metrics[f'{name}_percentile_{p}'] = deciles[i]

                except Exception as e:
                    warnings.warn(f"GPU Error calculating {name} centrality: {e}. Check cuGraph documentation for parameters.")
                    # Optional: Fallback to CPU? Could be slow.
                    # For now, just report None for aggregates if GPU fails.
                    metrics[f'{name}_mean'] = None
                    metrics[f'{name}_median'] = None
                    metrics[f'{name}_std'] = None
                    metrics[f'{name}_max'] = None
                    metrics[f'{name}_min'] = None
                    metrics[f'{name}_gini'] = None
                    for p in range(0, 101, 10):
                        metrics[f'{name}_percentile_{p}'] = None

        else:
            # --- CPU Path (NetworkX) ---
            warnings.warn("Using CPU (NetworkX) for centrality calculations.")
            g = self.graph
            for name, func in centrality_funcs_nx.items():
                try:
                    # NetworkX functions return dictionaries directly
                    if name == 'eigenvector':
                        values = func(g, max_iter=1000, tol=1e-3) # Add parameters for stability
                    elif name == 'pagerank':
                        values = func(g, alpha=0.85)
                    else:
                        values = func(g)

                    # Store node-level results if requested
                    if include_nodes:
                        centrality_values[name] = values
                
                    # Calculate aggregate statistics
                    values_array = np.array(list(values.values()))
                    if len(values_array) == 0: continue # Skip if no values
                
                    metrics[f'{name}_mean'] = np.mean(values_array)
                    metrics[f'{name}_median'] = np.median(values_array)
                    metrics[f'{name}_std'] = np.std(values_array)
                    metrics[f'{name}_max'] = np.max(values_array)
                    metrics[f'{name}_min'] = np.min(values_array)
                
                    # Calculate Gini coefficient using Numba GPU acceleration if available
                    if NUMBA_CUDA_AVAILABLE:
                        warnings.warn(f"Using Numba CUDA for Gini coefficient calculation of {name} centrality.")
                        metrics[f'{name}_gini'] = compute_gini_gpu(values_array)
                    else:
                        # Traditional method
                        values_array_sorted = np.sort(values_array)
                        n_vals = len(values_array_sorted)
                        if n_vals > 0 and np.sum(values_array_sorted) != 0:
                            index = np.arange(1, n_vals + 1)
                            metrics[f'{name}_gini'] = (2 * np.sum(index * values_array_sorted) / (n_vals * np.sum(values_array_sorted))) - (n_vals + 1) / n_vals
                        else:
                            metrics[f'{name}_gini'] = 0 if n_vals > 0 else None # Gini is 0 if all values are same, None if empty

                    # Calculate deciles
                    deciles = [np.percentile(values_array, p) for p in range(0, 101, 10)]
                    for i, p in enumerate(range(0, 101, 10)):
                        metrics[f'{name}_percentile_{p}'] = deciles[i]
                    
                except Exception as e:
                    warnings.warn(f"CPU Error calculating {name} centrality: {e}")
                    metrics[f'{name}_mean'] = None
                    metrics[f'{name}_median'] = None
                    metrics[f'{name}_std'] = None
                    metrics[f'{name}_max'] = None
                    metrics[f'{name}_min'] = None
                    metrics[f'{name}_gini'] = None
                    for p in range(0, 101, 10):
                        metrics[f'{name}_percentile_{p}'] = None
                
        # Include node values if requested (collected from either GPU or CPU path)
        if include_nodes:
            metrics['node_centrality'] = centrality_values
            
        return metrics
        
    def community_metrics(self, methods=None):
        """
        Calcula métricas de comunidades para o grafo.
        
        Parameters
        ----------
        methods : list, optional
            Lista de métodos de detecção de comunidades a usar
            
        Returns
        -------
        dict
            Dicionário com as métricas de comunidades
        """
        g = self.graph
        metrics = {}
        communities_results = {} # To store partition data if needed elsewhere
        
        # Default methods
        if methods is None:
            methods = ['louvain', 'label_propagation']
            
        # Check availability of CPU Louvain
        try:
            import community as community_louvain
            has_cpu_louvain = True
        except ImportError:
            has_cpu_louvain = False
        
        # --- Select execution path ---
        use_gpu_for_community = self.use_gpu
        G_cu = None
        if use_gpu_for_community:
            G_cu = self._get_cugraph_graph()
            if G_cu is None:
                use_gpu_for_community = False # Fallback
        
        # --- Community Detection ---
        for method in methods:
            partition = None
            modularity = None
            num_communities = None
            community_sizes = None
            
            try:
                if use_gpu_for_community:
                    # --- GPU Path ---
                    if method == 'louvain':
                        warnings.warn("Using GPU (cuGraph) for Louvain.")
                        result_df, modularity = cugraph.louvain(G_cu)
                    elif method == 'label_propagation':
                        warnings.warn("Using GPU (cuGraph) for Label Propagation.")
                        result_df, modularity = cugraph.label_propagation(G_cu, max_iter=100) # Use modularity score from Lבל
                        # Note: cuGraph LP returns partition, not modularity directly. We might need to calculate it.
                        # For now, let's assume modularity is computed if available, or use the partition.
                        # cuGraph's LP doesn't directly return modularity. Let's calculate it.
                        partition_map = dict(zip(result_df['vertex'].to_pandas(), result_df['labels'].to_pandas()))
                        # Need to unrenumber partition map keys if graph was renumbered
                        if G_cu.renumbered:
                            renumbered_nodes = result_df['vertex'].to_pandas()
                            original_nodes = G_cu.unrenumber(result_df['vertex'], 'vertex').to_pandas()
                            renumber_map = dict(zip(renumbered_nodes, original_nodes))
                            partition_map_orig = {renumber_map[k]: v for k, v in partition_map.items() if k in renumber_map}
                        else:
                            partition_map_orig = partition_map

                        # Calculate modularity using networkx on the original graph with the GPU partition
                        modularity = nx.community.modularity(self.graph, [{node for node, comm_id in partition_map_orig.items() if comm_id == c} for c in set(partition_map_orig.values())])

                    else:
                        warnings.warn(f"GPU community method '{method}' not implemented in this script, skipping.")
                        continue

                    # Process cuGraph results (common for Louvain/LP)
                    if 'result_df' in locals() and result_df is not None:
                        # Map back from renumbered vertices if needed
                        if G_cu.renumbered:
                            result_df = G_cu.unrenumber(result_df, 'vertex')

                        partition = dict(zip(result_df['vertex'].to_pandas(), result_df['labels'].to_pandas()))
                        communities_results[method] = partition # Store partition

                        # Calculate metrics from partition
                        num_communities = result_df['labels'].nunique()
                        community_sizes = result_df['labels'].value_counts().to_pandas().values

                else:
                    # --- CPU Path ---
                    if method == 'louvain':
                        if has_cpu_louvain:
                            warnings.warn("Using CPU (python-louvain) for Louvain.")
                            partition = community_louvain.best_partition(g)
                            modularity = community_louvain.modularity(partition, g)
                        else:
                            warnings.warn("CPU Louvain not available (install python-louvain), skipping.")
                            continue
                    elif method == 'label_propagation':
                        warnings.warn("Using CPU (NetworkX) for Label Propagation.")
                        communities_generator = nx.algorithms.community.label_propagation_communities(g)
                        communities_list = list(communities_generator)
                        partition = {}
                        for i, comm in enumerate(communities_list):
                            for node in comm:
                                partition[node] = i
                        # Calculate modularity using nx
                        modularity = nx.community.modularity(g, communities_list)

                    elif method == 'girvan_newman':
                        if self.n <= 1000: # Only run on small graphs
                            warnings.warn("Using CPU (NetworkX) for Girvan-Newman.")
                            from networkx.algorithms.community import girvan_newman
                            comp = girvan_newman(g)
                            communities_list = next(comp) # Take first level
                            partition = {}
                            for i, comm in enumerate(communities_list):
                                for node in comm:
                                    partition[node] = i
                            modularity = nx.community.modularity(g, communities_list)
                        else:
                            warnings.warn("Skipping Girvan-Newman on CPU for large graph (n > 1000).")
                            continue
                    else:
                        warnings.warn(f"CPU community method '{method}' not supported, skipping.")
                        continue

                    # Process CPU results (common)
                    if partition:
                        communities_results[method] = partition
                        community_to_nodes = {}
                        for node, comm_id in partition.items():
                            community_to_nodes.setdefault(comm_id, []).append(node)
                        num_communities = len(community_to_nodes)
                        community_sizes = [len(nodes) for nodes in community_to_nodes.values()]

                # --- Aggregate metrics (common for GPU/CPU if successful) ---
                if num_communities is not None and community_sizes is not None:
                    metrics[f'{method}_num_communities'] = num_communities
                    if modularity is not None:
                        # Ensure modularity is a standard float
                        if hasattr(modularity, 'item'):
                            modularity = modularity.item()
                        metrics[f'{method}_modularity'] = modularity
                    metrics[f'{method}_avg_community_size'] = np.mean(community_sizes)
                    metrics[f'{method}_max_community_size'] = np.max(community_sizes)
                    metrics[f'{method}_min_community_size'] = np.min(community_sizes)
                    metrics[f'{method}_community_size_std'] = np.std(community_sizes)

            except Exception as e:
                warnings.warn(f"Error during {method} community detection ({'GPU' if use_gpu_for_community else 'CPU'}): {e}")
                metrics[f'{method}_num_communities'] = None
                metrics[f'{method}_modularity'] = None
                metrics[f'{method}_avg_community_size'] = None
                metrics[f'{method}_max_community_size'] = None
                metrics[f'{method}_min_community_size'] = None
                metrics[f'{method}_community_size_std'] = None
                
        # Store community partitions if needed (e.g., for visualization)
        # metrics['communities'] = communities_results # Removed as per original logic (too verbose)
        
        return metrics
        
    def resilience_metrics(self, attack_strategies=None, steps=10):
        """
        Calcula métricas de resiliência para o grafo.
        
        Parameters
        ----------
        attack_strategies : list, optional
            Lista de estratégias de ataque a usar
        steps : int, optional
            Número de etapas para simulação de ataques
            
        Returns
        -------
        dict
            Dicionário com as métricas de resiliência
        """
        warnings.warn(f"Resilience metrics using {'GPU (CuPy)' if self.use_cupy else 'CPU (NumPy)'} for array operations.")
        g = self.graph.copy()
        metrics = {}
        xp = self.xp  # Use CuPy if available or NumPy otherwise
        
        # Default attack strategies
        if attack_strategies is None:
            attack_strategies = ['random', 'degree']
        
        # --- Basic Connectivity (NetworkX) ---
        # cuGraph k_core / core_number could be related but not direct replacement
        try:
            metrics['node_connectivity'] = nx.node_connectivity(g)
        except Exception as e:
            warnings.warn(f"Could not calculate node connectivity: {e}")
            metrics['node_connectivity'] = None
            
        try:
            metrics['edge_connectivity'] = nx.edge_connectivity(g)
        except Exception as e:
            warnings.warn(f"Could not calculate edge connectivity: {e}")
            metrics['edge_connectivity'] = None
            
        try:
            metrics['average_node_connectivity'] = nx.average_node_connectivity(g)
        except Exception as e:
             warnings.warn(f"Could not calculate average node connectivity: {e}")
             metrics['average_node_connectivity'] = None
        
        # --- Simulate Attacks (NetworkX + CuPy/NumPy) ---
        n_original = g.number_of_nodes()
        if n_original == 0: return metrics # Skip if graph is empty

        for strategy in attack_strategies:
            attack_g = g.copy() # Use a fresh copy for each strategy
            nodes_to_remove = []

            try:
                 if strategy == 'random':
                     nodes_to_remove = list(attack_g.nodes())
                     # Use CuPy's random module if available
                     if self.use_cupy:
                         # Convert to array, shuffle, convert back to list
                         nodes_array = xp.array(nodes_to_remove)
                         xp.random.shuffle(nodes_array)
                         nodes_to_remove = nodes_array.tolist()
                     else:
                         np.random.shuffle(nodes_to_remove)
                 elif strategy == 'degree':
                     # Use degree centrality calculation (respects GPU if enabled, though rest is CPU)
                     centrality_result = self.centrality_metrics(include_nodes=True)
                     if 'node_centrality' in centrality_result and 'degree' in centrality_result['node_centrality']:
                         degree_dict = centrality_result['node_centrality']['degree']
                         nodes_to_remove = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)
                     else: # Fallback to nx degree if centrality failed
                          degree_dict = dict(attack_g.degree())
                          nodes_to_remove = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)

                 elif strategy == 'betweenness':
                     # Use betweenness centrality calculation (respects GPU if enabled)
                     centrality_result = self.centrality_metrics(include_nodes=True)
                     if 'node_centrality' in centrality_result and 'betweenness' in centrality_result['node_centrality']:
                         betweenness_dict = centrality_result['node_centrality']['betweenness']
                         nodes_to_remove = sorted(betweenness_dict.keys(), key=lambda x: betweenness_dict[x], reverse=True)
                     else: # Fallback to nx betweenness if centrality failed
                          betweenness_dict = nx.betweenness_centrality(attack_g)
                          nodes_to_remove = sorted(betweenness_dict.keys(), key=lambda x: betweenness_dict[x], reverse=True)
                 else:
                     warnings.warn(f"Unsupported attack strategy: {strategy}")
                     continue

                 if not nodes_to_remove: continue # Skip if no nodes to remove

                 # Calculate number of nodes to remove at each step
                 num_to_remove_total = len(nodes_to_remove)
                 nodes_per_step = max(1, num_to_remove_total // steps)

                 # Metrics to track during attack
                 lcc_sizes_fraction = [] # Track fraction of original size

                 # Perform the attack in steps
                 for i in range(steps + 1): # Include initial state (step 0) and final state
                      # Calculate fraction of nodes removed
                      nodes_removed_count = min(i * nodes_per_step, num_to_remove_total)
                      fraction_removed = nodes_removed_count / n_original

                      # Calculate metrics *before* removal for this step
                      if attack_g.number_of_nodes() > 0:
                          # Size of largest connected component
                          largest_cc = max(nx.connected_components(attack_g), key=len, default=set())
                          lcc_size = len(largest_cc)
                          lcc_sizes_fraction.append(lcc_size / n_original)
                      else:
                          lcc_sizes_fraction.append(0)

                      # Stop if we've reached the end or removed all nodes
                      if i == steps or nodes_removed_count == num_to_remove_total:
                          break

                      # Determine nodes to remove *in this step*
                      start_idx = i * nodes_per_step
                      end_idx = min((i + 1) * nodes_per_step, num_to_remove_total)
                      step_nodes = nodes_to_remove[start_idx:end_idx]

                      # Remove nodes for the *next* iteration
                      attack_g.remove_nodes_from(step_nodes)


                 # Calculate area under the curve (AUC) for LCC size fraction
                 # Using fraction removed (0 to ~1) as x-axis
                 x_axis = xp.linspace(0, fraction_removed, len(lcc_sizes_fraction)) # Actual fraction removed at each measured point
                 lcc_sizes_array = xp.array(lcc_sizes_fraction)
                 
                 if len(x_axis) > 1:
                     if NUMBA_CUDA_AVAILABLE:
                         warnings.warn(f"Using Numba CUDA for AUC calculation in {strategy} resilience simulation.")
                         # Convert to numpy arrays for Numba (if they're CuPy arrays)
                         if hasattr(x_axis, 'get'):
                             x_axis_np = x_axis.get()
                         else:
                             x_axis_np = x_axis
                             
                         if hasattr(lcc_sizes_array, 'get'):
                             lcc_sizes_np = lcc_sizes_array.get()
                         else:
                             lcc_sizes_np = lcc_sizes_array
                             
                         lcc_auc = compute_trapz_gpu(x_axis_np, lcc_sizes_np)
                     elif self.use_cupy:
                         # CuPy's trapz function
                         lcc_auc = xp.trapz(lcc_sizes_array, x=x_axis)
                         # Convert to numpy/float if it's a cupy array
                         if hasattr(lcc_auc, 'get'):
                             lcc_auc = lcc_auc.get()
                     else:
                         lcc_auc = np.trapz(lcc_sizes_array, x=x_axis)
                 elif len(x_axis) == 1:
                     lcc_auc = lcc_sizes_fraction[0] # AUC is just the initial value if only one point
                 else:
                      lcc_auc = 0


                 # Add metrics to results (only AUC as per original logic)
                 metrics[f'{strategy}_attack_lcc_auc'] = lcc_auc

            except Exception as e:
                 warnings.warn(f"Error during {strategy} resilience simulation: {e}")
                 metrics[f'{strategy}_attack_lcc_auc'] = None

        return metrics


class SpatialMetrics(GraphMetrics):
    """
    Métricas especializadas para grafos geoespaciais.
    
    Esta classe estende GraphMetrics com métricas específicas para
    redes geoespaciais e análise de redes de transporte.
    Requires NetworkX graph with node/edge attributes. GPU acceleration
    applied to base graph metrics where applicable. Spatial specific metrics
    currently remain on CPU.
    """
    
    def __init__(self, geograph, use_gpu=False):
        """
        Inicializa o calculador de métricas espaciais.
        
        Parameters
        ----------
        geograph : FeatureGeoGraph
            Grafo geoespacial a ser analisado (must have node/edge attributes)
        use_gpu : bool, optional
            Se True, tenta usar cuGraph para aceleração de métricas base (default: False)
        """
        if not isinstance(geograph, FeatureGeoGraph):
             raise ValueError("Input must be a FeatureGeoGraph for SpatialMetrics")
        super().__init__(geograph, use_gpu=use_gpu)
        self.geograph = geograph
        
    def transportation_metrics(self, length_attribute='length', distance_attribute='distance'):
        """
        Calcula métricas específicas para redes de transporte.
        Currently uses NetworkX. Requires edges to have 'length' attribute.
        Accessibility calculation uses 'distance' attribute by default.
        
        Parameters
        ----------
        length_attribute : str, optional
             Name of the edge attribute containing the length/physical distance.
        distance_attribute : str, optional
             Name of the edge attribute containing the travel distance/cost for accessibility.
        
        Returns
        ------
        dict
            Dicionário com métricas de redes de transporte
        """
        warnings.warn("Transportation metrics currently use CPU (NetworkX).")
        g = self.graph
        metrics = {}
        if self.n == 0: return metrics # Handle empty graph

        # Basic transportation indices
        metrics['beta_index'] = self.m / self.n
        if self.n >= 3:
            max_edges_planar = 3 * self.n - 6 # Max edges in planar graph
            metrics['gamma_index'] = self.m / max_edges_planar if max_edges_planar > 0 else 0
            # Calculate alpha index
            try:
                p = nx.number_connected_components(g) # Recalculate components for accuracy
            except:
                p = 1 # Assume 1 component if calculation fails
            num_cycles = self.m - self.n + p
            max_cycles = 2 * self.n - 5 # Max cycles in planar graph
            metrics['alpha_index'] = num_cycles / max_cycles if max_cycles > 0 else 0
            metrics['cyclomatic_number'] = num_cycles
        else:
            metrics['gamma_index'] = 0
            metrics['alpha_index'] = 0
            metrics['cyclomatic_number'] = 0

        # Eta index and total length (Requires length attribute)
        total_length = 0
        num_edges_with_length = 0
        try:
             for u, v, data in g.edges(data=True):
                 length = data.get(length_attribute) # Use specified attribute
                 if length is not None:
                     try:
                          total_length += float(length)
                          num_edges_with_length += 1
                     except (ValueError, TypeError):
                          warnings.warn(f"Non-numeric value found for edge attribute '{length_attribute}' between {u}-{v}. Skipping edge for length calculation.")

             metrics['eta_index'] = total_length / num_edges_with_length if num_edges_with_length > 0 else None
             metrics['total_network_length'] = total_length if num_edges_with_length > 0 else None
        except Exception as e:
             warnings.warn(f"Error calculating Eta index/Total length using attribute '{length_attribute}': {e}")
             metrics['eta_index'] = None
             metrics['total_network_length'] = None

        # Pi index and network diameter (Requires distance attribute and APSP)
        # NetworkX APSP can be slow.
        max_dist = None
        avg_dist = None # Calculate average shortest path distance too
        total_dist_sum = 0
        num_paths = 0

        try:
            # Check if weight attribute exists
            has_distance = all(distance_attribute in data for _, _, data in g.edges(data=True))
            if not has_distance:
                 warnings.warn(f"Distance attribute '{distance_attribute}' not found on all edges. Skipping Pi index and distance-based diameter.")
                 raise ValueError(f"Missing distance attribute '{distance_attribute}'")

            # Use all_pairs_dijkstra for both path length and paths if needed
            # Note: This is very memory intensive for large graphs!
            if nx.is_connected(g):
                 shortest_paths_lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight=distance_attribute))
                 max_dist = 0
                 for source in shortest_paths_lengths:
                      for target, dist in shortest_paths_lengths[source].items():
                           if dist > max_dist:
                                max_dist = dist
                           total_dist_sum += dist
                           num_paths += 1
                 avg_dist = total_dist_sum / num_paths if num_paths > 0 else 0
            else:
                 # Calculate on largest component only for disconnected graphs
                 warnings.warn("Graph is disconnected. Calculating diameter/path length on largest component only.")
                 largest_cc = max(nx.connected_components(g), key=len)
                 subgraph = g.subgraph(largest_cc)
                 if subgraph.number_of_nodes() > 1:
                     shortest_paths_lengths = dict(nx.all_pairs_dijkstra_path_length(subgraph, weight=distance_attribute))
                     max_dist = 0
                     for source in shortest_paths_lengths:
                          for target, dist in shortest_paths_lengths[source].items():
                               if dist > max_dist:
                                    max_dist = dist
                               total_dist_sum += dist
                               num_paths += 1
                     avg_dist = total_dist_sum / num_paths if num_paths > 0 else 0
                 else:
                     max_dist = 0
                     avg_dist = 0

            metrics['network_diameter_distance'] = max_dist
            metrics['avg_shortest_path_distance'] = avg_dist # Added metric
            metrics['pi_index'] = total_length / max_dist if max_dist is not None and max_dist > 0 and total_length is not None else None

        except Exception as e:
            warnings.warn(f"Error calculating Pi index/Diameter (distance) using attribute '{distance_attribute}': {e}")
            metrics['network_diameter_distance'] = None
            metrics['avg_shortest_path_distance'] = None
            metrics['pi_index'] = None

        return metrics
    
    def spatial_autocorrelation(self, attribute, global_only=True):
        """
        Calcula métricas de autocorrelação espacial para um atributo.
        Uses NetworkX and NumPy/SciPy. Requires node attribute.
        
        Parameters
        ----------
        attribute : str
            Nome do atributo do nó para calcular autocorrelação
        global_only : bool, optional
            Indica se deve retornar apenas métricas globais (Moran's I, Geary's C)
        
        Returns
        ------
        dict
            Dicionário com métricas de autocorrelação espacial
        """
        warnings.warn(f"Spatial autocorrelation using {'GPU (CuPy)' if self.use_cupy else 'CPU (NumPy)'} for matrix operations.")
        g = self.graph
        metrics = {}
        xp = self.xp  # Use CuPy if available or NumPy otherwise
        
        if self.n == 0: return metrics # Handle empty graph

        # Extract attribute values
        values = {}
        nodes_with_attr = []
        for node, attrs in g.nodes(data=True):
            if attribute in attrs:
                try:
                    values[node] = float(attrs[attribute])
                    nodes_with_attr.append(node)
                except (ValueError, TypeError):
                     warnings.warn(f"Non-numeric value for attribute '{attribute}' on node {node}. Skipping node.")
            else:
                 warnings.warn(f"Attribute '{attribute}' not found on node {node}. Skipping node.")

        if len(nodes_with_attr) < 2: # Need at least 2 nodes with data
             metrics['error'] = f'Attribute {attribute} not found or not numeric on sufficient nodes ({len(nodes_with_attr)} found)'
             return metrics

        # Work with the subgraph containing only nodes with the attribute
        subgraph = g.subgraph(nodes_with_attr)
        n = subgraph.number_of_nodes()
        
        # Convert to appropriate array type (NumPy or CuPy)
        value_array = xp.array([values[node] for node in nodes_with_attr]) # Ensure order matches nodelist
        
        # Get adjacency matrix
        adj_matrix_np = nx.to_numpy_array(subgraph, nodelist=nodes_with_attr) # Use nodelist to ensure order
        
        # Convert to CuPy if available
        if self.use_cupy:
            adj_matrix = xp.array(adj_matrix_np)
        else:
            adj_matrix = adj_matrix_np
            
        w_sum = xp.sum(adj_matrix) # Sum of weights (assuming binary adjacency here)
        
        # Convert to standard float if necessary
        if hasattr(w_sum, 'get'):
            w_sum = w_sum.get()

        if w_sum == 0:
            metrics['error'] = 'No edges in the subgraph with the specified attribute'
            return metrics

        # --- Global Moran's I ---
        try:
             z = value_array - xp.mean(value_array)
             z_squared_sum = xp.sum(z**2)
             
             # Convert to standard float if necessary
             if hasattr(z_squared_sum, 'get'):
                 z_squared_sum = z_squared_sum.get()
                 
             if z_squared_sum == 0: # Handle case where all values are the same
                 morans_i = 0 # Or undefined, depends on interpretation
             else:
                  # Element-wise product (outer product)
                  z_matrix = xp.outer(z, z)
                  moran_numerator = xp.sum(z_matrix * adj_matrix)
                  
                  # Convert to standard float if necessary
                  if hasattr(moran_numerator, 'get'):
                      moran_numerator = moran_numerator.get()
                      
                  morans_i = (n / w_sum) * (moran_numerator / z_squared_sum)

             metrics['morans_i'] = morans_i
             metrics['expected_i'] = -1.0 / (n - 1) if n > 1 else None
        except Exception as e:
             warnings.warn(f"Error calculating Moran's I for attribute '{attribute}': {e}")
             metrics['morans_i'] = None
             metrics['expected_i'] = None

        # --- Global Geary's C ---
        try:
             z = value_array - xp.mean(value_array) # Recalculate z just in case
             z_squared_sum = xp.sum(z**2)
             
             # Convert to standard float if necessary
             if hasattr(z_squared_sum, 'get'):
                 z_squared_sum = z_squared_sum.get()
                 
             if z_squared_sum == 0:
                 gearys_c = 1 # Or undefined, Geary's C tends to 1 if no variation
             else:
                 # Calculate sum of squared differences for neighbors
                 # This is computationally intensive, good candidate for GPU
                 if self.use_cupy:
                     # Vectorized operations on GPU
                     # Create matrices of values (broadcast)
                     vals_i = value_array.reshape(-1, 1)  # Column vector
                     vals_j = value_array.reshape(1, -1)  # Row vector
                     
                     # Compute squared differences
                     diff_sq_matrix = (vals_i - vals_j)**2
                     
                     # Multiply by adjacency matrix and sum
                     diff_sq_sum = xp.sum(adj_matrix * diff_sq_matrix)
                     
                     # Convert to standard float if necessary
                     if hasattr(diff_sq_sum, 'get'):
                         diff_sq_sum = diff_sq_sum.get()
                 else:
                     # Standard CPU implementation
                     diff_sq_sum = 0
                     for i in range(n):
                         for j in range(n):
                              if adj_matrix[i, j] > 0: # Check if neighbors
                                  diff_sq_sum += adj_matrix[i, j] * ((value_array[i] - value_array[j])**2)

                 geary_numerator = (n - 1) * diff_sq_sum
                 geary_denominator = 2 * w_sum * z_squared_sum
                 gearys_c = geary_numerator / geary_denominator if geary_denominator != 0 else None

             metrics['gearys_c'] = gearys_c
             metrics['expected_c'] = 1.0
        except Exception as e:
             warnings.warn(f"Error calculating Geary's C for attribute '{attribute}': {e}")
             metrics['gearys_c'] = None
             metrics['expected_c'] = None

        # --- Getis-Ord General G (Requires non-negative values) ---
        if xp.all(value_array >= 0):
            try:
                # Sum of products for neighbors
                # This is also computationally intensive
                if self.use_cupy:
                    # Vectorized version on GPU
                    vals_i = value_array.reshape(-1, 1)  # Column vector
                    vals_j = value_array.reshape(1, -1)  # Row vector
                    
                    # Create product matrix
                    prod_matrix = vals_i * vals_j
                    
                    # Zero out diagonal (exclude self)
                    eye_matrix = xp.eye(n)
                    adj_matrix_no_diag = adj_matrix * (1 - eye_matrix)
                    
                    # Multiply by adjacency matrix and sum
                    prod_sum = xp.sum(prod_matrix * adj_matrix_no_diag)
                    
                    # Convert to standard float if necessary
                    if hasattr(prod_sum, 'get'):
                        prod_sum = prod_sum.get()
                else:
                    # Standard CPU implementation
                    prod_sum = 0
                    for i in range(n):
                        for j in range(n):
                            if i != j and adj_matrix[i, j] > 0: # Neighbors, excluding self
                                prod_sum += adj_matrix[i, j] * value_array[i] * value_array[j]

                # Sum of all pairs (excluding self)
                total_sum = xp.sum(value_array)
                
                # Convert to standard float if necessary
                if hasattr(total_sum, 'get'):
                    total_sum = total_sum.get()
                    
                total_sum_sq = total_sum**2
                sum_sq = xp.sum(value_array**2)
                
                # Convert to standard float if necessary
                if hasattr(sum_sq, 'get'):
                    sum_sq = sum_sq.get()
                    
                all_pairs_sum = total_sum_sq - sum_sq

                getis_ord_g = prod_sum / all_pairs_sum if all_pairs_sum != 0 else None
                metrics['getis_ord_g'] = getis_ord_g
                metrics['expected_g'] = w_sum / (n * (n - 1)) if n > 1 else None
            except Exception as e:
                warnings.warn(f"Error calculating Getis-Ord G for attribute '{attribute}': {e}")
                metrics['getis_ord_g'] = None
                metrics['expected_g'] = None
        else:
            warnings.warn(f"Skipping Getis-Ord G for attribute '{attribute}' as it contains negative values.")

        # --- Local Metrics (if requested) ---
        if not global_only:
             warnings.warn("Local spatial autocorrelation metrics are not implemented in this version.")
             # Placeholder for future implementation if needed
             metrics['local_morans_i'] = None
             metrics['local_getis_ord_g'] = None

        return metrics
    
    def accessibility_metrics(self, weight='distance'):
        """
        Calcula métricas de acessibilidade (baseadas em closeness centrality).
        Uses NetworkX all_pairs_dijkstra_path_length which can be slow.
        Requires edges to have the specified weight attribute.
        
        Parameters
        ----------
        weight : str, optional
            Atributo de aresta a ser usado como peso/custo (e.g., 'distance', 'time')
        
        Returns
        ------
        dict
            Dicionário com métricas de acessibilidade (agregadas e por nó)
        """
        warnings.warn(f"Accessibility metrics currently use CPU (NetworkX) with weight='{weight}'.")
        g = self.graph
        metrics = {}
        node_accessibility = {}
        if self.n == 0: return metrics

        try:
            # Check if weight attribute exists
            has_weight = all(weight in data for _, _, data in g.edges(data=True))
            if not has_weight:
                warnings.warn(f"Weight attribute '{weight}' not found on all edges. Accessibility calculation might fail or produce incorrect results.")
                 # Proceed, but results might be based on topological distance if Dijkstra falls back

            # Compute closeness centrality using the specified weight
            # NetworkX closeness centrality handles disconnected graphs internally
            closeness_dict = nx.closeness_centrality(g, distance=weight)

            # Store node-level accessibility
            node_accessibility = closeness_dict

            # Calculate aggregate statistics
            closeness_values = list(closeness_dict.values())
            if not closeness_values: # Handle empty result
                 raise ValueError("Closeness centrality calculation returned empty results.")

            metrics['accessibility_mean'] = np.mean(closeness_values)
            metrics['accessibility_median'] = np.median(closeness_values)
            metrics['accessibility_std'] = np.std(closeness_values)
            metrics['accessibility_min'] = np.min(closeness_values)
            metrics['accessibility_max'] = np.max(closeness_values)

            # Calculate accessibility Gini coefficient
            closeness_sorted = np.sort(closeness_values)
            n_vals = len(closeness_sorted)
            if n_vals > 0 and np.sum(closeness_sorted) != 0:
                index = np.arange(1, n_vals + 1)
                metrics['accessibility_gini'] = (2 * np.sum(index * closeness_sorted) / (n_vals * np.sum(closeness_sorted))) - (n_vals + 1) / n_vals
            else:
                 metrics['accessibility_gini'] = 0 if n_vals > 0 else None

            # Calculate accessibility deciles
            deciles = [np.percentile(closeness_values, p) for p in range(0, 101, 10)]
            for i, p in enumerate(range(0, 101, 10)):
                metrics[f'accessibility_percentile_{p}'] = deciles[i]

            # Add node-level data
            metrics['node_accessibility'] = node_accessibility # Already a dict

        except Exception as e:
            warnings.warn(f"Error calculating accessibility metrics with weight '{weight}': {e}")
            metrics['error'] = str(e)
            # Set aggregates to None
            metrics['accessibility_mean'] = None
            metrics['accessibility_median'] = None
            # ... etc ...
            metrics['node_accessibility'] = None

        return metrics
        
    def compute_all_spatial_metrics(self, node_attributes_autocorr=None,
                                    transport_length_attr='length',
                                    transport_distance_attr='distance',
                                    accessibility_weight='distance'):
        """
        Computa todas as métricas espaciais e base disponíveis.
        Respects the use_gpu flag for base GraphMetrics.
        
        Parameters
        ----------
        node_attributes_autocorr : list, optional
            Lista de atributos de nó para calcular autocorrelação espacial
        transport_length_attr : str, optional
             Edge attribute for physical length in transportation metrics.
        transport_distance_attr : str, optional
             Edge attribute for travel distance/cost in transportation metrics.
        accessibility_weight : str, optional
             Edge attribute for accessibility calculation weight.
        
        Returns
        ------
        dict
            Dicionário com todas as métricas
        """
        warnings.warn(f"Computing all spatial metrics (GPU enabled for base metrics: {self.use_gpu})")
        # Start with base metrics (respects self.use_gpu)
        metrics = super().compute_all_metrics()

        # Add transportation metrics (CPU)
        try:
            metrics.update(self.transportation_metrics(length_attribute=transport_length_attr, distance_attribute=transport_distance_attr))
        except Exception as e:
            metrics['transportation_error'] = str(e)
            warnings.warn(f"Error in transportation_metrics: {e}")

        # Add accessibility metrics (CPU)
        try:
            # Remove node-level accessibility from final dict unless specifically requested?
            # For now, keep it as returned by the function.
            access_metrics = self.accessibility_metrics(weight=accessibility_weight)
            if 'node_accessibility' in access_metrics:
                 # Maybe store node accessibility separately or remove for summary?
                 # For now, keep it.
                 pass
            metrics.update(access_metrics)

        except Exception as e:
            metrics['accessibility_error'] = str(e)
            warnings.warn(f"Error in accessibility_metrics: {e}")

        # Add spatial autocorrelation for specified attributes (CPU)
        if node_attributes_autocorr:
            for attr in node_attributes_autocorr:
                try:
                    autocorr = self.spatial_autocorrelation(attr, global_only=True) # Only global for summary

                    # Add results with attribute prefix
                    prefix = f'{attr}_'
                    for k, v in autocorr.items():
                        if k != 'error':
                            metrics[prefix + k] = v
                        else:
                            metrics[prefix + 'autocorr_error'] = v
                            warnings.warn(f"Autocorrelation error for '{attr}': {v}")

                except Exception as e:
                    error_key = f'{attr}_autocorrelation_error'
                    metrics[error_key] = str(e)
                    warnings.warn(f"Error calculating spatial autocorrelation for '{attr}': {e}")

        return metrics


class MultiLayerMetrics:
    """
    Métricas para análise de grafos multicamada.
    Currently uses NetworkX and basic analysis. GPU acceleration not implemented.

    Esta classe fornece métodos para calcular métricas específicas
    para grafos multicamada, incluindo métricas de interdependência
    entre camadas e correlação.
    """

    def __init__(self, multi_graph, use_gpu_per_layer=False):
        """
        Inicializa o calculador de métricas multicamada.

        Parameters
        ----------
        multi_graph : MultiLayerFeatureGraph
            Grafo multicamada a ser analisado
        use_gpu_per_layer : bool, optional
             If True, use GPU acceleration when analyzing individual layers (default: False)
        """
        warnings.warn("MultiLayerMetrics currently uses CPU (NetworkX) for inter-layer analysis.")
        if not isinstance(multi_graph, MultiLayerFeatureGraph):
             raise ValueError("Input must be a MultiLayerFeatureGraph")

        self.multi_graph = multi_graph
        self.layer_names = multi_graph.get_layer_names()
        self.use_gpu_per_layer = use_gpu_per_layer # Store flag

        # Calculate metrics for each layer individually
        self.layer_metrics = {}
        for layer_name in self.layer_names:
             try:
                 graph = self.multi_graph.get_layer(layer_name) # Gets FeatureGeoGraph
                 # Use SpatialMetrics for individual layers, passing the GPU flag
                 spatial_metrics_calculator = SpatialMetrics(graph, use_gpu=self.use_gpu_per_layer)
                 self.layer_metrics[layer_name] = spatial_metrics_calculator.compute_all_spatial_metrics()
             except Exception as e:
                  warnings.warn(f"Failed to compute metrics for layer '{layer_name}': {e}")
                  self.layer_metrics[layer_name] = {'error': str(e)}


    def layer_comparison_metrics(self):
        """
        Calcula métricas comparativas entre as camadas.
        Compares metrics already calculated in __init__.

        Returns
        ------
        dict
            Dicionário com métricas comparativas
        """
        metrics = {}
        num_layers = len(self.layer_names)
        if num_layers < 2:
            return metrics # Need at least two layers to compare

        # Compare basic metrics across layers (if available)
        basic_metric_keys = ['num_nodes', 'num_edges', 'density', 'avg_degree', 'avg_clustering']
        for key in basic_metric_keys:
             values = [self.layer_metrics[name].get(key) for name in self.layer_names if self.layer_metrics[name].get(key) is not None]
             if len(values) >= 2:
                 metrics[f'{key}_mean_across_layers'] = np.mean(values)
                 metrics[f'{key}_std_across_layers'] = np.std(values)
                 metrics[f'{key}_range_across_layers'] = np.ptp(values) # Peak-to-peak (max-min)

        # Compare centrality distributions (e.g., Gini coefficient)
        centrality_gini_keys = [k for k in self.layer_metrics[self.layer_names[0]].keys() if k.endswith('_gini')]
        for key in centrality_gini_keys:
            values = [self.layer_metrics[name].get(key) for name in self.layer_names if self.layer_metrics[name].get(key) is not None]
            if len(values) >= 2:
                 metrics[f'{key}_mean_across_layers'] = np.mean(values)
                 metrics[f'{key}_std_across_layers'] = np.std(values)

        # Jaccard similarity of node sets between layers
        node_sets = {name: set(self.multi_graph.get_layer(name).nodes()) for name in self.layer_names}
        for i in range(num_layers):
             for j in range(i + 1, num_layers):
                 name1, name2 = self.layer_names[i], self.layer_names[j]
                 set1, set2 = node_sets[name1], node_sets[name2]
                 intersection = len(set1.intersection(set2))
                 union = len(set1.union(set2))
                 jaccard = intersection / union if union > 0 else 0
                 metrics[f'node_jaccard_{name1}_{name2}'] = jaccard

        # TODO: Add more sophisticated comparison metrics (e.g., spectral distance, layer entanglement)

        return metrics


    def interlayer_edge_metrics(self):
        """
        Calcula métricas relacionadas às arestas entre camadas.

        Returns
        ------
        dict
            Dicionário com métricas das arestas entre camadas
        """
        metrics = {}
        interlayer_edges = self.multi_graph.get_interlayer_edges()

        if interlayer_edges is None or interlayer_edges.empty:
            metrics['num_interlayer_edges'] = 0
            metrics['interlayer_density'] = 0
            return metrics

        # Basic counts
        num_interlayer = len(interlayer_edges)
        metrics['num_interlayer_edges'] = num_interlayer

        # Calculate potential number of interlayer edges (depends on node overlap)
        max_potential_interlayer = 0
        node_sets = {name: set(self.multi_graph.get_layer(name).nodes()) for name in self.layer_names}
        num_layers = len(self.layer_names)
        for i in range(num_layers):
             for j in range(i + 1, num_layers):
                 name1, name2 = self.layer_names[i], self.layer_names[j]
                 set1, set2 = node_sets[name1], node_sets[name2]
                 # Potential edges between overlapping nodes
                 overlap_nodes = set1.intersection(set2)
                 # This calculation might need refinement based on specific multi-layer definition
                 # Simplified: assume edge possible between any node in layer i and any in layer j
                 max_potential_interlayer += len(set1) * len(set2)


        metrics['interlayer_density'] = num_interlayer / max_potential_interlayer if max_potential_interlayer > 0 else 0

        # Analyze interlayer edge attributes (if any)
        if 'weight' in interlayer_edges.columns:
            weights = interlayer_edges['weight'].dropna()
            if not weights.empty:
                metrics['interlayer_weight_mean'] = weights.mean()
                metrics['interlayer_weight_median'] = weights.median()
                metrics['interlayer_weight_std'] = weights.std()
                metrics['interlayer_weight_min'] = weights.min()
                metrics['interlayer_weight_max'] = weights.max()

        # Connectivity provided by interlayer edges
        # Count nodes involved in interlayer edges
        interlayer_nodes = set(interlayer_edges['source']).union(set(interlayer_edges['target']))
        metrics['num_nodes_with_interlayer_edges'] = len(interlayer_nodes)

        # Calculate degree for interlayer edges only
        source_degrees = interlayer_edges['source'].value_counts()
        target_degrees = interlayer_edges['target'].value_counts()
        interlayer_degree = source_degrees.add(target_degrees, fill_value=0) # Combine degrees

        if not interlayer_degree.empty:
             metrics['interlayer_degree_mean'] = interlayer_degree.mean()
             metrics['interlayer_degree_median'] = interlayer_degree.median()
             metrics['interlayer_degree_std'] = interlayer_degree.std()
             metrics['interlayer_degree_max'] = interlayer_degree.max()

        # TODO: Add metrics like interlayer betweenness contribution

        return metrics


    def multi_layer_centrality(self):
        """
        Calcula métricas de centralidade considerando a estrutura multicamada.
        Placeholder for advanced multi-layer centrality measures. Currently returns empty.

        Returns
        ------
        dict
            Dicionário com métricas de centralidade multicamada
        """
        warnings.warn("Advanced multi-layer centrality metrics are not yet implemented.")
        metrics = {}
        # TODO: Implement multi-layer degree, betweenness, PageRank etc.
        # Requires libraries like PyMultiplex or custom implementations.

        # Example: Multi-layer degree (sum of degrees across layers for overlapping nodes)
        # multi_degree = {}
        # all_nodes = self.multi_graph.get_all_nodes()
        # for node in all_nodes:
        #     total_degree = 0
        #     for layer_name in self.layer_names:
        #         layer_graph = self.multi_graph.get_layer(layer_name)
        #         if layer_graph.has_node(node):
        #              total_degree += layer_graph.degree(node)
        #     # Add interlayer degree if available
        #     # ... calculation needed ...
        #     multi_degree[node] = total_degree

        # metrics['multi_degree_mean'] = np.mean(list(multi_degree.values()))
        # ... etc ...

        return metrics


    def layer_interdependence(self):
        """
        Calcula métricas de interdependência entre camadas.
        Placeholder for advanced interdependence analysis. Currently returns empty.

        Returns
        ------
        dict
            Dicionário com métricas de interdependência
        """
        warnings.warn("Layer interdependence metrics are not yet implemented.")
        metrics = {}
        # TODO: Implement metrics like:
        # - Correlation of centrality measures between layers for overlapping nodes
        # - Impact of node removal in one layer on connectivity/metrics in another layer
        # - Layer entanglement / Redundancy

        # Example: Correlate degree centrality between two layers
        # if len(self.layer_names) >= 2:
        #     name1, name2 = self.layer_names[0], self.layer_names[1]
        #     metrics1 = self.layer_metrics.get(name1, {})
        #     metrics2 = self.layer_metrics.get(name2, {})
        #
        #     if 'node_centrality' in metrics1 and 'node_centrality' in metrics2:
        #         cent1 = metrics1['node_centrality'].get('degree', {})
        #         cent2 = metrics2['node_centrality'].get('degree', {})
        #
        #         overlapping_nodes = set(cent1.keys()).intersection(set(cent2.keys()))
        #         if len(overlapping_nodes) > 1:
        #              vals1 = [cent1[n] for n in overlapping_nodes]
        #              vals2 = [cent2[n] for n in overlapping_nodes]
        #              correlation, p_value = stats.pearsonr(vals1, vals2)
        #              metrics[f'degree_corr_{name1}_{name2}'] = correlation
        #              metrics[f'degree_corr_pvalue_{name1}_{name2}'] = p_value

        return metrics


    def compute_all_multilayer_metrics(self):
        """
        Computa todas as métricas multicamada disponíveis.

        Returns
        ------
        dict
            Dicionário com todas as métricas multicamada agregadas
        """
        metrics = {"layer_metrics": self.layer_metrics} # Include per-layer metrics

        try:
            metrics.update(self.layer_comparison_metrics())
        except Exception as e:
            metrics['layer_comparison_error'] = str(e)
            warnings.warn(f"Error in layer_comparison_metrics: {e}")

        try:
            metrics.update(self.interlayer_edge_metrics())
        except Exception as e:
            metrics['interlayer_edge_error'] = str(e)
            warnings.warn(f"Error in interlayer_edge_metrics: {e}")

        try:
            metrics.update(self.multi_layer_centrality())
        except Exception as e:
            metrics['multi_layer_centrality_error'] = str(e)
            warnings.warn(f"Error in multi_layer_centrality: {e}")

        try:
            metrics.update(self.layer_interdependence())
        except Exception as e:
            metrics['layer_interdependence_error'] = str(e)
            warnings.warn(f"Error in layer_interdependence: {e}")

        return metrics 

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" GRAPHO TERRAIN - MÉTRICAS DE GRAFOS ".center(80, "="))
    print("="*80 + "\n")
    
    # Verificar a disponibilidade da GPU
    print("Verificando disponibilidade de aceleração por GPU...")
    if RAPIDS_AVAILABLE:
        print("√ RAPIDS (cuGraph, cuDF) disponível para algoritmos de grafos.")
    else:
        print("× RAPIDS (cuGraph, cuDF) não disponível.")
        print("  - Para instalar: pip install cugraph-cu12x cudf-cu12x")
    
    if CUPY_AVAILABLE:
        print("√ CuPy disponível para operações matriciais.")
        print(f"  - CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    else:
        print("× CuPy não disponível ou não funcionando corretamente.")
        print("  - Para instalar: pip install cupy-cuda12x")
        
    if NUMBA_CUDA_AVAILABLE:
        print("√ Numba CUDA disponível para operações específicas.")
        print(f"  - CUDA Compute Capability: {cuda.get_current_device().compute_capability}")
    else:
        print("× Numba CUDA não disponível.")
        print("  - Certifique-se de que o Numba está instalado e o CUDA está acessível.")
    
    print("\nCriando grafo de exemplo e calculando métricas...")
    # Criar um grafo aleatório maior para demonstrar o benefício da GPU
    n_nodes = 150
    p_edge = 0.03
    G = nx.erdos_renyi_graph(n=n_nodes, p=p_edge)
    print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    
    # Usar GPU para aceleração se disponível
    use_gpu = RAPIDS_AVAILABLE or CUPY_AVAILABLE or NUMBA_CUDA_AVAILABLE
    
    print(f"Usando modo {'híbrido GPU/CPU' if use_gpu else 'apenas CPU'} para cálculos...")
    
    # Medir o tempo de execução para demonstrar o benefício
    import time
    start_time = time.time()
    
    try:
        # Inicializar o calculador de métricas
        graph_metrics = GraphMetrics(G, use_gpu=use_gpu)
        
        # Calcular métricas usando o método híbrido
        print("Calculando métricas com estratégia híbrida (este processo pode demorar)...")
        all_metrics = graph_metrics.hybrid_compute_all_metrics()
        
        # Tempo de execução
        execution_time = time.time() - start_time
        
        print(f"\nMétricas computadas em {execution_time:.2f} segundos.")
        
        # Exibir métricas básicas
        print("\nMétricas Básicas:")
        basic_metrics = ['num_nodes', 'num_edges', 'density', 'avg_degree', 'avg_clustering', 
                         'diameter', 'avg_path_length', 'connected_components', 'largest_cc_fraction']
        for metric in basic_metrics:
            if metric in all_metrics:
                print(f"  {metric}: {all_metrics[metric]}")
                
        # Exibir métricas de comunidade
        print("\nMétricas de Comunidade:")
        community_metrics = [k for k in all_metrics.keys() if 'community' in k or 'modularity' in k]
        for metric in sorted(community_metrics):
            if all_metrics[metric] is not None:
                print(f"  {metric}: {all_metrics[metric]}")
            
        # Exibir métricas de resiliência
        print("\nMétricas de Resiliência:")
        resilience_metrics = [k for k in all_metrics.keys() if 'attack' in k or 'connectivity' in k]
        for metric in sorted(resilience_metrics):
            if all_metrics[metric] is not None:
                print(f"  {metric}: {all_metrics[metric]}")
        
        # Dicas para melhoria de desempenho
        if not use_gpu:
            print("\nDicas para melhorar o desempenho:")
            print("1. Instale o CuPy para aceleração de operações matriciais:")
            print("   pip install cupy-cuda12x")
            print("2. Para grafos maiores, considere instalar o RAPIDS para algoritmos de grafos:")
            print("   pip install cugraph-cu12x cudf-cu12x")
            print("3. Para operações específicas, instale o Numba com suporte a CUDA:")
            print("   pip install numba")
            print("4. Certifique-se de que suas DLLs do CUDA estão acessíveis ao Python")
        else:
            if not RAPIDS_AVAILABLE:
                print("\nDicas para melhorar ainda mais o desempenho:")
                print("1. Instale o RAPIDS para acelerar algoritmos específicos de grafos:")
                print("   pip install cugraph-cu12x cudf-cu12x")
                
            if not NUMBA_CUDA_AVAILABLE and not CUPY_AVAILABLE:
                print("2. Instale o Numba ou CuPy para aceleração de operações matriciais")
            
    except Exception as e:
        print(f"\nErro ao executar pipeline de métricas: {e}")
        import traceback
        traceback.print_exc()