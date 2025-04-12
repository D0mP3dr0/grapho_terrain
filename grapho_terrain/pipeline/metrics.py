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

class GraphMetrics:
    """
    Classe para calcular métricas avançadas de grafos.
    
    Esta classe fornece métodos para calcular métricas de grafos,
    incluindo métricas básicas, centralidade, comunidades e resiliência.
    """
    
    def __init__(self, graph):
        """
        Inicializa o calculador de métricas.
        
        Parameters
        ----------
        graph : networkx.Graph, FeatureGeoGraph
            Grafo a ser analisado
        """
        if isinstance(graph, FeatureGeoGraph):
            self.graph = graph.to_networkx()
        else:
            self.graph = graph
            
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()
        
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
            metrics['avg_clustering'] = nx.average_clustering(g)
        except:
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
        
    def centrality_metrics(self, normalized=True, include_nodes=False):
        """
        Calcula métricas de centralidade para o grafo.
        
        Parameters
        ----------
        normalized : bool, optional
            Indica se as centralidades devem ser normalizadas
        include_nodes : bool, optional
            Indica se deve retornar os valores por nó ou apenas agregações
            
        Returns
        -------
        dict
            Dicionário com as métricas de centralidade
        """
        g = self.graph
        metrics = {}
        
        # Calculate centrality measures
        centrality_funcs = {
            'degree': nx.degree_centrality,
            'betweenness': nx.betweenness_centrality,
            'closeness': nx.closeness_centrality
        }
        
        # Try to calculate eigenvector centrality (may fail for some graphs)
        try:
            centrality_funcs['eigenvector'] = nx.eigenvector_centrality
        except:
            pass
            
        # Try to calculate pagerank (may fail for some graphs)
        try:
            centrality_funcs['pagerank'] = nx.pagerank
        except:
            pass
        
        # Calculate each centrality measure
        centrality_values = {}
        for name, func in centrality_funcs.items():
            try:
                values = func(g)
                centrality_values[name] = values
                
                # Add aggregate statistics
                values_array = np.array(list(values.values()))
                
                metrics[f'{name}_mean'] = np.mean(values_array)
                metrics[f'{name}_median'] = np.median(values_array)
                metrics[f'{name}_std'] = np.std(values_array)
                metrics[f'{name}_max'] = np.max(values_array)
                metrics[f'{name}_min'] = np.min(values_array)
                
                # Add histogram metrics (deciles)
                deciles = [np.percentile(values_array, p) for p in range(0, 101, 10)]
                for i, p in enumerate(range(0, 101, 10)):
                    metrics[f'{name}_percentile_{p}'] = deciles[i]
                    
                # Calculate Gini coefficient (inequality measure)
                values_array = np.sort(values_array)
                n = len(values_array)
                index = np.arange(1, n + 1)
                metrics[f'{name}_gini'] = (2 * np.sum(index * values_array) / (n * np.sum(values_array))) - (n + 1) / n
            except Exception as e:
                print(f"Error calculating {name} centrality: {e}")
                
        # Include node values if requested
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
        
        # Default methods
        if methods is None:
            methods = ['louvain', 'label_propagation']
            
        # Try to import community detection algorithms
        try:
            import community as community_louvain
            has_louvain = True
        except ImportError:
            has_louvain = False
        
        # Detect communities using different methods
        communities = {}
        
        # Louvain method
        if 'louvain' in methods and has_louvain:
            try:
                partition = community_louvain.best_partition(g)
                communities['louvain'] = partition
                
                # Get community assignments
                community_to_nodes = {}
                for node, comm_id in partition.items():
                    if comm_id not in community_to_nodes:
                        community_to_nodes[comm_id] = []
                    community_to_nodes[comm_id].append(node)
                
                # Calculate metrics
                num_communities = len(community_to_nodes)
                community_sizes = [len(nodes) for nodes in community_to_nodes.values()]
                
                metrics['louvain_num_communities'] = num_communities
                metrics['louvain_modularity'] = community_louvain.modularity(partition, g)
                metrics['louvain_avg_community_size'] = np.mean(community_sizes)
                metrics['louvain_max_community_size'] = np.max(community_sizes)
                metrics['louvain_min_community_size'] = np.min(community_sizes)
                metrics['louvain_community_size_std'] = np.std(community_sizes)
            except Exception as e:
                print(f"Error in Louvain community detection: {e}")
        
        # Label propagation
        if 'label_propagation' in methods:
            try:
                communities_generator = nx.algorithms.community.label_propagation_communities(g)
                communities_list = list(communities_generator)
                
                # Convert to node -> community format
                partition = {}
                for i, comm in enumerate(communities_list):
                    for node in comm:
                        partition[node] = i
                        
                communities['label_propagation'] = partition
                
                # Calculate metrics
                num_communities = len(communities_list)
                community_sizes = [len(comm) for comm in communities_list]
                
                metrics['label_propagation_num_communities'] = num_communities
                metrics['label_propagation_avg_community_size'] = np.mean(community_sizes)
                metrics['label_propagation_max_community_size'] = np.max(community_sizes)
                metrics['label_propagation_min_community_size'] = np.min(community_sizes)
                metrics['label_propagation_community_size_std'] = np.std(community_sizes)
            except Exception as e:
                print(f"Error in Label Propagation community detection: {e}")
                
        # Girvan-Newman (only for smaller graphs due to computational complexity)
        if 'girvan_newman' in methods and self.n <= 1000:
            try:
                from networkx.algorithms.community import girvan_newman
                comp = girvan_newman(g)
                
                # Take the first level of the hierarchy
                communities_list = next(comp)
                communities_list = [list(c) for c in communities_list]
                
                # Convert to node -> community format
                partition = {}
                for i, comm in enumerate(communities_list):
                    for node in comm:
                        partition[node] = i
                        
                communities['girvan_newman'] = partition
                
                # Calculate metrics
                num_communities = len(communities_list)
                community_sizes = [len(comm) for comm in communities_list]
                
                metrics['girvan_newman_num_communities'] = num_communities
                metrics['girvan_newman_avg_community_size'] = np.mean(community_sizes)
                metrics['girvan_newman_max_community_size'] = np.max(community_sizes)
                metrics['girvan_newman_min_community_size'] = np.min(community_sizes)
                metrics['girvan_newman_community_size_std'] = np.std(community_sizes)
            except Exception as e:
                print(f"Error in Girvan-Newman community detection: {e}")
                
        # Add communities to the metrics if requested
        metrics['communities'] = communities
        
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
        g = self.graph.copy()
        metrics = {}
        
        # Default attack strategies
        if attack_strategies is None:
            attack_strategies = ['random', 'degree']
        
        # Basic connectivity metrics
        try:
            metrics['node_connectivity'] = nx.node_connectivity(g)
        except:
            metrics['node_connectivity'] = None
            
        try:
            metrics['edge_connectivity'] = nx.edge_connectivity(g)
        except:
            metrics['edge_connectivity'] = None
            
        try:
            metrics['average_node_connectivity'] = nx.average_node_connectivity(g)
        except:
            metrics['average_node_connectivity'] = None
        
        # Simulate attacks on the network
        for strategy in attack_strategies:
            # Create a copy of the graph
            attack_g = g.copy()
            n_original = attack_g.number_of_nodes()
            
            # List of nodes to remove
            if strategy == 'random':
                nodes_to_remove = list(attack_g.nodes())
                np.random.shuffle(nodes_to_remove)
            elif strategy == 'degree':
                degree_dict = dict(attack_g.degree())
                nodes_to_remove = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)
            elif strategy == 'betweenness':
                betweenness_dict = nx.betweenness_centrality(attack_g)
                nodes_to_remove = sorted(betweenness_dict.keys(), key=lambda x: betweenness_dict[x], reverse=True)
            else:
                continue
                
            # Calculate number of nodes to remove at each step
            nodes_per_step = max(1, len(nodes_to_remove) // steps)
            
            # Metrics to track during attack
            lcc_sizes = []
            num_components = []
            
            # Perform the attack in steps
            for i in range(0, len(nodes_to_remove), nodes_per_step):
                # Get nodes to remove in this step
                step_nodes = nodes_to_remove[i:i+nodes_per_step]
                
                # Remove nodes
                attack_g.remove_nodes_from(step_nodes)
                
                # Calculate metrics after removal
                if attack_g.number_of_nodes() > 0:
                    # Size of largest connected component
                    largest_cc = max(nx.connected_components(attack_g), key=len)
                    lcc_size = len(largest_cc) / n_original
                    lcc_sizes.append(lcc_size)
                    
                    # Number of connected components
                    num_components.append(nx.number_connected_components(attack_g))
                else:
                    lcc_sizes.append(0)
                    num_components.append(0)
                    
            # Calculate area under the curve (AUC) for LCC size
            # This is a measure of robustness - higher values mean better robustness
            lcc_auc = np.trapz(lcc_sizes, dx=1.0/steps)
            
            # Add metrics to results
            metrics[f'{strategy}_attack_lcc_auc'] = lcc_auc
            metrics[f'{strategy}_attack_lcc_sizes'] = lcc_sizes
            metrics[f'{strategy}_attack_num_components'] = num_components
            
        return metrics
    
    def compute_all_metrics(self, include_communities=True, include_resilience=True):
        """
        Computa todas as métricas disponíveis para o grafo.
        
        Parameters
        ----------
        include_communities : bool, optional
            Indica se deve calcular métricas de comunidade
        include_resilience : bool, optional
            Indica se deve calcular métricas de resiliência
            
        Returns
        -------
        dict
            Dicionário com todas as métricas
        """
        metrics = {}
        
        # Compute basic metrics
        metrics.update(self.basic_metrics())
        
        # Compute centrality metrics
        metrics.update(self.centrality_metrics())
        
        # Compute community metrics if requested
        if include_communities:
            community_metrics = self.community_metrics()
            
            # Remove the 'communities' key (too verbose for summary)
            if 'communities' in community_metrics:
                del community_metrics['communities']
                
            metrics.update(community_metrics)
            
        # Compute resilience metrics if requested
        if include_resilience:
            resilience_metrics = self.resilience_metrics()
            
            # Remove attack trajectory arrays (too verbose for summary)
            for key in list(resilience_metrics.keys()):
                if isinstance(resilience_metrics[key], list) and len(resilience_metrics[key]) > 0:
                    del resilience_metrics[key]
                    
            metrics.update(resilience_metrics)
            
        return metrics


class SpatialMetrics(GraphMetrics):
    """
    Métricas especializadas para grafos geoespaciais.
    
    Esta classe estende GraphMetrics com métricas específicas para
    redes geoespaciais e análise de redes de transporte.
    """
    
    def __init__(self, geograph):
        """
        Inicializa o calculador de métricas espaciais.
        
        Parameters
        ----------
        geograph : FeatureGeoGraph
            Grafo geoespacial a ser analisado
        """
        super().__init__(geograph)
        self.geograph = geograph
        
    def transportation_metrics(self):
        """
        Calcula métricas específicas para redes de transporte.
        
        Returns
        -------
        dict
            Dicionário com métricas de redes de transporte
        """
        g = self.graph
        metrics = {}
        
        # Basic transportation indices
        # Beta index: average number of edges per node
        metrics['beta_index'] = self.m / self.n if self.n > 0 else 0
        
        # Gamma index: ratio of actual number of edges to the maximum possible number
        # For a planar graph, max number of edges = 3*n - 6 (if n >= 3)
        if self.n >= 3:
            max_edges = 3 * self.n - 6
            metrics['gamma_index'] = self.m / max_edges if max_edges > 0 else 0
        else:
            metrics['gamma_index'] = 0
            
        # Alpha index: ratio of actual number of cycles to the maximum possible number
        # Number of cycles = m - n + p, where p is the number of connected components
        p = nx.number_connected_components(g)
        num_cycles = self.m - self.n + p
        max_cycles = 2 * self.n - 5 if self.n >= 3 else 0
        metrics['alpha_index'] = num_cycles / max_cycles if max_cycles > 0 else 0
        
        # Cyclomatic number: number of fundamental cycles
        metrics['cyclomatic_number'] = num_cycles
        
        # Eta index: average edge length (if edges have 'length' attribute)
        total_length = 0
        num_edges_with_length = 0
        
        for u, v, data in g.edges(data=True):
            if 'length' in data:
                total_length += data['length']
                num_edges_with_length += 1
                
        metrics['eta_index'] = total_length / num_edges_with_length if num_edges_with_length > 0 else None
        metrics['total_network_length'] = total_length
        
        # Pi index: ratio of total network length to the diameter (measured in distance)
        try:
            # Get shortest paths
            shortest_paths = dict(nx.all_pairs_dijkstra_path_length(g, weight='length'))
            
            # Find the maximum shortest path (diameter in terms of distance)
            max_dist = 0
            for source in shortest_paths:
                for target, dist in shortest_paths[source].items():
                    max_dist = max(max_dist, dist)
                    
            metrics['network_diameter_distance'] = max_dist
            metrics['pi_index'] = total_length / max_dist if max_dist > 0 else None
        except:
            metrics['network_diameter_distance'] = None
            metrics['pi_index'] = None
            
        return metrics
    
    def spatial_autocorrelation(self, attribute, global_only=True):
        """
        Calcula métricas de autocorrelação espacial para um atributo.
        
        Parameters
        ----------
        attribute : str
            Nome do atributo para calcular autocorrelação
        global_only : bool, optional
            Indica se deve retornar apenas métricas globais
            
        Returns
        -------
        dict
            Dicionário com métricas de autocorrelação espacial
        """
        g = self.graph
        metrics = {}
        
        # Extract attribute values for all nodes
        try:
            values = {}
            for node, attrs in g.nodes(data=True):
                if attribute in attrs:
                    values[node] = float(attrs[attribute])
                    
            if not values:
                return {'error': f'Attribute {attribute} not found or not numeric'}
                
            # Convert to numpy array and get adjacency matrix
            nodes = list(values.keys())
            value_array = np.array([values[node] for node in nodes])
            
            # Create adjacency matrix
            adj_matrix = nx.to_numpy_array(g, nodelist=nodes)
            
            # Compute global Moran's I
            n = len(nodes)
            w_sum = np.sum(adj_matrix)
            
            if w_sum == 0:
                return {'error': 'No edges in the graph'}
                
            # Standardize values
            z = value_array - np.mean(value_array)
            z_squared_sum = np.sum(z**2)
            
            # Compute Moran's I
            moran_numerator = np.sum(np.outer(z, z) * adj_matrix) / w_sum
            moran_denominator = z_squared_sum / n
            
            morans_i = moran_numerator / moran_denominator
            metrics['morans_i'] = morans_i
            
            # Expected value of Moran's I under randomization
            expected_i = -1.0 / (n - 1)
            metrics['expected_i'] = expected_i
            
            # Compute Geary's C
            geary_numerator = (n - 1) * np.sum(np.subtract.outer(z, z)**2 * adj_matrix) / (2 * w_sum)
            geary_denominator = z_squared_sum / n
            
            gearys_c = geary_numerator / geary_denominator
            metrics['gearys_c'] = gearys_c
            
            # Expected value of Geary's C under randomization
            expected_c = 1.0
            metrics['expected_c'] = expected_c
            
            # Compute Getis-Ord General G
            if np.all(value_array >= 0):
                g_numerator = np.sum(np.outer(value_array, value_array) * adj_matrix)
                g_denominator = np.sum(value_array)**2 - np.sum(value_array**2)
                
                getis_ord_g = g_numerator / g_denominator if g_denominator != 0 else None
                metrics['getis_ord_g'] = getis_ord_g
                
                # Expected value of General G under randomization
                expected_g = np.sum(adj_matrix) / (n * (n - 1))
                metrics['expected_g'] = expected_g
                
            # Compute local metrics if requested
            if not global_only:
                # Local Moran's I
                local_i = {}
                for i, node in enumerate(nodes):
                    weighted_sum = 0
                    for j, neighbor in enumerate(nodes):
                        if adj_matrix[i, j] > 0:
                            weighted_sum += adj_matrix[i, j] * z[j]
                            
                    local_i[node] = (z[i] / z_squared_sum) * n * weighted_sum
                    
                metrics['local_morans_i'] = local_i
                
                # Local Getis-Ord G
                if np.all(value_array >= 0):
                    local_g = {}
                    for i, node in enumerate(nodes):
                        weighted_sum = 0
                        for j, neighbor in enumerate(nodes):
                            if adj_matrix[i, j] > 0:
                                weighted_sum += adj_matrix[i, j] * value_array[j]
                                
                        local_g[node] = weighted_sum / (np.sum(value_array) - value_array[i])
                        
                    metrics['local_getis_ord_g'] = local_g
        except Exception as e:
            metrics['error'] = str(e)
            
        return metrics
    
    def accessibility_metrics(self, weight='distance'):
        """
        Calcula métricas de acessibilidade para o grafo.
        
        Parameters
        ----------
        weight : str, optional
            Atributo de aresta a ser usado como peso
            
        Returns
        -------
        dict
            Dicionário com métricas de acessibilidade
        """
        g = self.graph
        metrics = {}
        
        try:
            # Compute shortest path lengths using the specified weight
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight=weight))
            
            # Compute accessibility metrics for each node
            closeness = {}
            # ... centrality também é uma medida de acessibilidade
            for node in g.nodes():
                if node in path_lengths:
                    # Sum of distances to all other nodes
                    distances_sum = sum(path_lengths[node].values())
                    
                    # Number of reachable nodes
                    reachable = len(path_lengths[node])
                    
                    # Compute closeness (inverse of average distance)
                    if distances_sum > 0 and reachable > 1:
                        closeness[node] = (reachable - 1) / distances_sum
                    else:
                        closeness[node] = 0
                else:
                    closeness[node] = 0
                    
            # Calculate statistics for closeness
            closeness_values = list(closeness.values())
            
            metrics['accessibility_mean'] = np.mean(closeness_values)
            metrics['accessibility_median'] = np.median(closeness_values)
            metrics['accessibility_std'] = np.std(closeness_values)
            metrics['accessibility_min'] = np.min(closeness_values)
            metrics['accessibility_max'] = np.max(closeness_values)
            
            # Calculate accessibility Gini coefficient (inequality)
            closeness_sorted = np.sort(closeness_values)
            n = len(closeness_sorted)
            index = np.arange(1, n + 1)
            metrics['accessibility_gini'] = (2 * np.sum(index * closeness_sorted) / (n * np.sum(closeness_sorted))) - (n + 1) / n
            
            # Calculate accessibility deciles
            deciles = [np.percentile(closeness_values, p) for p in range(0, 101, 10)]
            for i, p in enumerate(range(0, 101, 10)):
                metrics[f'accessibility_percentile_{p}'] = deciles[i]
                
            # Add node-level data if needed
            metrics['node_accessibility'] = closeness
        except Exception as e:
            metrics['error'] = str(e)
            
        return metrics
        
    def compute_all_spatial_metrics(self, node_attributes=None):
        """
        Computa todas as métricas espaciais disponíveis.
        
        Parameters
        ----------
        node_attributes : list, optional
            Lista de atributos para calcular autocorrelação espacial
            
        Returns
        -------
        dict
            Dicionário com todas as métricas espaciais
        """
        # Start with base metrics
        metrics = super().compute_all_metrics()
        
        # Add transportation metrics
        metrics.update(self.transportation_metrics())
        
        # Add accessibility metrics
        try:
            metrics.update(self.accessibility_metrics())
        except Exception as e:
            metrics['accessibility_error'] = str(e)
        
        # Add spatial autocorrelation for specified attributes
        if node_attributes:
            for attr in node_attributes:
                try:
                    autocorr = self.spatial_autocorrelation(attr)
                    
                    for k, v in autocorr.items():
                        if k != 'error':
                            metrics[f'{attr}_{k}'] = v
                except Exception as e:
                    metrics[f'{attr}_autocorrelation_error'] = str(e)
                    
        return metrics


class MultiLayerMetrics:
    """
    Métricas para análise de grafos multicamada.
    
    Esta classe fornece métodos para calcular métricas específicas
    para grafos multicamada, incluindo métricas de interdependência
    entre camadas e correlação.
    """
    
    def __init__(self, multi_graph):
        """
        Inicializa o calculador de métricas multicamada.
        
        Parameters
        ----------
        multi_graph : MultiLayerFeatureGraph
            Grafo multicamada a ser analisado
        """
        self.multi_graph = multi_graph
        self.layer_metrics = {}
        
        # Calculate metrics for each layer
        for layer_id, layer in multi_graph.layers.items():
            metrics = SpatialMetrics(layer)
            self.layer_metrics[layer_id] = metrics
            
    def layer_comparison_metrics(self):
        """
        Calcula métricas comparativas entre camadas.
        
        Returns
        -------
        dict
            Dicionário com métricas comparativas
        """
        metrics = {}
        
        # Calculate basic metrics for each layer
        layer_basic_metrics = {}
        for layer_id, metrics_calc in self.layer_metrics.items():
            layer_basic_metrics[layer_id] = metrics_calc.basic_metrics()
            
        # Add layer basic metrics with layer prefix
        for layer_id, layer_metrics in layer_basic_metrics.items():
            for metric_name, value in layer_metrics.items():
                metrics[f'{layer_id}_{metric_name}'] = value
                
        # Calculate comparison metrics
        layer_ids = list(self.layer_metrics.keys())
        
        for i, layer1 in enumerate(layer_ids):
            for j, layer2 in enumerate(layer_ids):
                if i < j:
                    # Compare basic topological properties
                    for metric_name in ['density', 'avg_degree', 'avg_clustering', 'avg_path_length']:
                        if metric_name in layer_basic_metrics[layer1] and metric_name in layer_basic_metrics[layer2]:
                            val1 = layer_basic_metrics[layer1][metric_name]
                            val2 = layer_basic_metrics[layer2][metric_name]
                            
                            if val1 is not None and val2 is not None:
                                # Calculate ratio
                                ratio = val1 / val2 if val2 != 0 else float('inf')
                                metrics[f'{layer1}_{layer2}_{metric_name}_ratio'] = ratio
                                
                                # Calculate difference
                                metrics[f'{layer1}_{layer2}_{metric_name}_diff'] = val1 - val2
                                
        return metrics
    
    def interlayer_edge_metrics(self):
        """
        Calcula métricas relacionadas às conexões entre camadas.
        
        Returns
        -------
        dict
            Dicionário com métricas de conexões entre camadas
        """
        metrics = {}
        
        # Count number of interlayer edges between each pair of layers
        for (layer1, layer2), edges in self.multi_graph.interlayer_edges.items():
            # Count edges
            edge_count = len(edges)
            metrics[f'{layer1}_{layer2}_edge_count'] = edge_count
            
            # Calculate edge density between layers
            n1 = len(self.multi_graph.layers[layer1].nodes)
            n2 = len(self.multi_graph.layers[layer2].nodes)
            
            max_edges = n1 * n2
            density = edge_count / max_edges if max_edges > 0 else 0
            metrics[f'{layer1}_{layer2}_edge_density'] = density
            
            # Calculate edge weight statistics
            if edges and 'weight' in edges[0]:
                weights = [e['weight'] for e in edges if 'weight' in e]
                
                metrics[f'{layer1}_{layer2}_edge_weight_mean'] = np.mean(weights)
                metrics[f'{layer1}_{layer2}_edge_weight_median'] = np.median(weights)
                metrics[f'{layer1}_{layer2}_edge_weight_std'] = np.std(weights)
                metrics[f'{layer1}_{layer2}_edge_weight_min'] = np.min(weights)
                metrics[f'{layer1}_{layer2}_edge_weight_max'] = np.max(weights)
                
            # Calculate degree distribution of interlayer edges
            node_degrees_layer1 = {}
            node_degrees_layer2 = {}
            
            for edge in edges:
                source = edge['source_node']
                target = edge['target_node']
                
                node_degrees_layer1[source] = node_degrees_layer1.get(source, 0) + 1
                node_degrees_layer2[target] = node_degrees_layer2.get(target, 0) + 1
                
            # Calculate degree statistics for layer1 nodes
            if node_degrees_layer1:
                degrees = list(node_degrees_layer1.values())
                
                metrics[f'{layer1}_to_{layer2}_degree_mean'] = np.mean(degrees)
                metrics[f'{layer1}_to_{layer2}_degree_median'] = np.median(degrees)
                metrics[f'{layer1}_to_{layer2}_degree_std'] = np.std(degrees)
                metrics[f'{layer1}_to_{layer2}_degree_max'] = np.max(degrees)
                
                # Calculate proportion of nodes with interlayer connections
                metrics[f'{layer1}_to_{layer2}_connected_fraction'] = len(node_degrees_layer1) / n1 if n1 > 0 else 0
                
            # Calculate degree statistics for layer2 nodes
            if node_degrees_layer2:
                degrees = list(node_degrees_layer2.values())
                
                metrics[f'{layer2}_from_{layer1}_degree_mean'] = np.mean(degrees)
                metrics[f'{layer2}_from_{layer1}_degree_median'] = np.median(degrees)
                metrics[f'{layer2}_from_{layer1}_degree_std'] = np.std(degrees)
                metrics[f'{layer2}_from_{layer1}_degree_max'] = np.max(degrees)
                
                # Calculate proportion of nodes with interlayer connections
                metrics[f'{layer2}_from_{layer1}_connected_fraction'] = len(node_degrees_layer2) / n2 if n2 > 0 else 0
                
        return metrics
    
    def multi_layer_centrality(self):
        """
        Calcula métricas de centralidade para o grafo multicamada.
        
        Returns
        -------
        dict
            Dicionário com métricas de centralidade multicamada
        """
        metrics = {}
        
        # Calculate centrality correlation between layers
        layer_ids = list(self.layer_metrics.keys())
        
        # For each pair of layers
        for i, layer1 in enumerate(layer_ids):
            for j, layer2 in enumerate(layer_ids):
                if i < j:
                    # Get centrality metrics
                    ctr1 = self.layer_metrics[layer1].centrality_metrics(include_nodes=True)
                    ctr2 = self.layer_metrics[layer2].centrality_metrics(include_nodes=True)
                    
                    # Skip if node_centrality not available
                    if 'node_centrality' not in ctr1 or 'node_centrality' not in ctr2:
                        continue
                        
                    # Find nodes that exist in both layers
                    for ctr_type in ['degree', 'betweenness', 'closeness']:
                        if ctr_type in ctr1['node_centrality'] and ctr_type in ctr2['node_centrality']:
                            # Get nodes that exist in both layers with interlayer edges
                            common_nodes = set()
                            
                            # Check interlayer edges from layer1 to layer2
                            layer_pair = (layer1, layer2)
                            if layer_pair in self.multi_graph.interlayer_edges:
                                for edge in self.multi_graph.interlayer_edges[layer_pair]:
                                    common_nodes.add((edge['source_node'], edge['target_node']))
                                    
                            # Skip if no common nodes with edges
                            if not common_nodes:
                                continue
                                
                            # Get centrality values for nodes in the respective layers
                            values1 = []
                            values2 = []
                            
                            for source, target in common_nodes:
                                if source in ctr1['node_centrality'][ctr_type] and target in ctr2['node_centrality'][ctr_type]:
                                    values1.append(ctr1['node_centrality'][ctr_type][source])
                                    values2.append(ctr2['node_centrality'][ctr_type][target])
                                    
                            # Skip if not enough values
                            if len(values1) < 3:
                                continue
                                
                            # Calculate correlation
                            pearson_corr, pearson_p = stats.pearsonr(values1, values2)
                            spearman_corr, spearman_p = stats.spearmanr(values1, values2)
                            
                            metrics[f'{layer1}_{layer2}_{ctr_type}_pearson'] = pearson_corr
                            metrics[f'{layer1}_{layer2}_{ctr_type}_pearson_p'] = pearson_p
                            metrics[f'{layer1}_{layer2}_{ctr_type}_spearman'] = spearman_corr
                            metrics[f'{layer1}_{layer2}_{ctr_type}_spearman_p'] = spearman_p
                            
        return metrics
    
    def layer_interdependence(self):
        """
        Calcula métricas de interdependência entre camadas.
        
        Returns
        -------
        dict
            Dicionário com métricas de interdependência
        """
        metrics = {}
        
        # Superlayer metrics
        # Create a single combined graph
        import networkx as nx
        superlayer = nx.Graph()
        
        # Add all nodes from all layers
        for layer_id, layer in self.multi_graph.layers.items():
            for node_id, node in layer.nodes.items():
                # Add node with layer info
                superlayer.add_node(f"{layer_id}_{node_id}", layer=layer_id, original_id=node_id)
                
            # Add all edges within layers
            for edge in layer.edges.values():
                superlayer.add_edge(
                    f"{layer_id}_{edge.source}", 
                    f"{layer_id}_{edge.target}",
                    weight=edge.weight,
                    intralayer=True,
                    layer=layer_id
                )
                
        # Add interlayer edges
        for (layer1, layer2), edges in self.multi_graph.interlayer_edges.items():
            for edge in edges:
                superlayer.add_edge(
                    f"{layer1}_{edge['source_node']}",
                    f"{layer2}_{edge['target_node']}",
                    weight=edge.get('weight', 1.0),
                    intralayer=False,
                    source_layer=layer1,
                    target_layer=layer2
                )
                
        # Calculate metrics on the superlayer
        if superlayer.number_of_nodes() > 0:
            # Number of nodes and edges
            metrics['superlayer_nodes'] = superlayer.number_of_nodes()
            metrics['superlayer_edges'] = superlayer.number_of_edges()
            
            # Count intralayer vs interlayer edges
            intralayer_edges = sum(1 for _, _, data in superlayer.edges(data=True) if data.get('intralayer', False))
            interlayer_edges = superlayer.number_of_edges() - intralayer_edges
            
            metrics['superlayer_intralayer_edges'] = intralayer_edges
            metrics['superlayer_interlayer_edges'] = interlayer_edges
            metrics['interlayer_to_intralayer_ratio'] = interlayer_edges / intralayer_edges if intralayer_edges > 0 else float('inf')
            
            # Node interdependence
            # Fraction of nodes with interlayer connections
            nodes_with_interlayer = set()
            
            for u, v, data in superlayer.edges(data=True):
                if not data.get('intralayer', True):
                    nodes_with_interlayer.add(u)
                    nodes_with_interlayer.add(v)
                    
            metrics['nodes_with_interlayer_connections'] = len(nodes_with_interlayer)
            metrics['nodes_with_interlayer_fraction'] = len(nodes_with_interlayer) / superlayer.number_of_nodes()
            
            # Calculate largest connected component
            if not nx.is_connected(superlayer):
                largest_cc = max(nx.connected_components(superlayer), key=len)
                metrics['superlayer_largest_cc_size'] = len(largest_cc)
                metrics['superlayer_largest_cc_fraction'] = len(largest_cc) / superlayer.number_of_nodes()
                
                # Check which layers are connected in the largest component
                layers_in_cc = set()
                for node in largest_cc:
                    node_data = superlayer.nodes[node]
                    layers_in_cc.add(node_data['layer'])
                    
                metrics['superlayer_layers_in_largest_cc'] = len(layers_in_cc)
                metrics['superlayer_fraction_layers_in_largest_cc'] = len(layers_in_cc) / len(self.multi_graph.layers)
                
        return metrics
    
    def compute_all_multilayer_metrics(self):
        """
        Computa todas as métricas multicamada disponíveis.
        
        Returns
        -------
        dict
            Dicionário com todas as métricas multicamada
        """
        metrics = {}
        
        # Layer comparison metrics
        metrics.update(self.layer_comparison_metrics())
        
        # Interlayer edge metrics
        metrics.update(self.interlayer_edge_metrics())
        
        # Multi-layer centrality
        metrics.update(self.multi_layer_centrality())
        
        # Layer interdependence
        metrics.update(self.layer_interdependence())
        
        return metrics 