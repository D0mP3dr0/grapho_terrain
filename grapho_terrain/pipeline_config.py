"""
Configuração do pipeline integrado para o projeto Grapho Terrain.

Este módulo define as configurações para um pipeline fluido e completo que integra:
1. Pré-processamento de camadas geoespaciais (do projeto UFABC)
2. Análise de cobertura de ERBs
3. Criação e métricas de grafos (do projeto MBA)
"""

# Configurações para pré-processamento de camadas
LAYER_PREPROCESS_CONFIG = {
    # Configurações gerais
    'output_directory': 'data/processed',
    'create_backup': True,
    'verbose': True,
    
    # Camadas a processar
    'layers': {
        'erbs': {
            'source': 'erbs',
            'numeric_features': [
                'potencia_watts', 'ganho_antena', 'frequencia_mhz', 
                'altura_m', 'eirp', 'raio_km'
            ],
            'categorical_features': [
                'operadora', 'tipo_area'
            ],
            'normalization': 'standard',  # standard, minmax, robust, none
            'handle_missing': 'mean',  # mean, median, mode, zero, drop
            'create_buffer': True,
            'buffer_distance': 0.01,  # em graus (aprox. 1km)
            'filter_outliers': True,
            'outlier_method': 'iqr',  # iqr, zscore, isolation_forest
            'outlier_threshold': 1.5
        },
        'edificacoes': {
            'source': 'buildings',
            'numeric_features': [
                'altura', 'num_pavimentos', 'area', 'perimetro'
            ],
            'categorical_features': [
                'tipo', 'uso', 'material'
            ],
            'normalization': 'minmax',
            'handle_missing': 'median',
            'simplify_geometry': True,
            'simplify_tolerance': 0.0001,
            'create_centroids': True,
            'filter_outliers': True,
            'outlier_method': 'zscore',
            'outlier_threshold': 3.0
        },
        'vias': {
            'source': 'roads',
            'numeric_features': [
                'comprimento', 'largura', 'velocidade_max', 'num_faixas'
            ],
            'categorical_features': [
                'tipo', 'pavimento', 'sentido'
            ],
            'normalization': 'minmax',
            'handle_missing': 'median',
            'simplify_geometry': True,
            'simplify_tolerance': 0.0001,
            'create_buffer': True,
            'buffer_distance': 0.0005,  # em graus (aprox. 50m)
            'filter_outliers': False
        },
        'hidrografia': {
            'source': 'water',
            'numeric_features': [
                'area', 'perimetro', 'profundidade'
            ],
            'categorical_features': [
                'tipo', 'regime'
            ],
            'normalization': 'minmax',
            'handle_missing': 'median',
            'simplify_geometry': True,
            'simplify_tolerance': 0.0001,
            'create_buffer': True,
            'buffer_distance': 0.0005,  # em graus (aprox. 50m)
            'filter_outliers': False
        },
        'uso_solo': {
            'source': 'land_use',
            'numeric_features': [
                'area', 'perimetro'
            ],
            'categorical_features': [
                'classe', 'densidade'
            ],
            'normalization': 'minmax',
            'handle_missing': 'mode',
            'simplify_geometry': True,
            'simplify_tolerance': 0.0001,
            'filter_outliers': False
        },
        'curvas_nivel': {
            'source': 'contours',
            'numeric_features': [
                'elevacao', 'comprimento'
            ],
            'categorical_features': [],
            'normalization': 'minmax',
            'handle_missing': 'median',
            'simplify_geometry': True,
            'simplify_tolerance': 0.0001,
            'filter_outliers': False
        }
    }
}

# Configurações para análise de cobertura de ERBs
ERB_COVERAGE_CONFIG = {
    # Parâmetros de cálculo de cobertura
    'frequencias': {
        '700': {'classe': 'baixa_freq', 'penetracao_indoor': 'alta'},
        '850': {'classe': 'baixa_freq', 'penetracao_indoor': 'alta'},
        '900': {'classe': 'baixa_freq', 'penetracao_indoor': 'alta'},
        '1800': {'classe': 'media_freq', 'penetracao_indoor': 'media'},
        '2100': {'classe': 'media_freq', 'penetracao_indoor': 'media'},
        '2600': {'classe': 'alta_freq', 'penetracao_indoor': 'baixa'},
        '3500': {'classe': 'alta_freq', 'penetracao_indoor': 'muito_baixa'},
        '26000': {'classe': 'mmWave', 'penetracao_indoor': 'extremamente_baixa'}
    },
    
    # Fatores de atenuação por tipo de ambiente
    'fatores_atenuacao': {
        'urbana_densa': 3.0,
        'urbana': 2.5,
        'suburbana': 2.0,
        'rural': 1.5,
        'aberta': 1.2
    },
    
    # Parâmetros para modelagem de setores
    'angulo_setor_padrao': 120,  # em graus
    'resolucao_setor': 36,  # número de pontos na borda do setor
    'tilt_padrao': 5,  # em graus
    
    # Parâmetros para análise de interferência
    'limiar_interferencia_db': -95,
    'considerar_interferencia': True,
    
    # Parâmetros para análises de visibilidade
    'considerar_visibilidade': True,
    'resolucao_dem': 30,  # em metros
    'considerar_difração': True,
    'considerar_reflexão': False,
    
    # Parâmetros para visualização
    'gerar_mapa_calor': True,
    'gerar_mapa_cobertura': True,
    'gerar_mapa_sobreposicao': True,
    'cores_operadoras': {
        'Operadora A': '#FF5555',
        'Operadora B': '#55FF55',
        'Operadora C': '#5555FF',
        'Operadora D': '#FFFF55',
        'Outra': '#FF55FF'
    },
    
    # Parâmetros de desempenho e processamento
    'batch_size': 100,
    'max_threads': 8,
    'use_gpu': True,
    'precisao_grid': 30  # em metros, resolução do grid de análise
}

# Configurações para criação e análise de grafos
GRAPH_CONFIG = {
    # Parâmetros para construção de grafos
    'graph_construction': {
        'k_nearest': 5,  # número de vizinhos mais próximos
        'max_distance': 5.0,  # em km, distância máxima para conexão
        'method': 'knn',  # knn, distance, delaunay, gabriel, beta_skeleton
        'beta': 1.0,  # para beta-skeleton
        'weighted': True,
        'directed': False
    },
    
    # Métricas a serem calculadas
    'graph_metrics': {
        'basic': [
            'num_nodes', 'num_edges', 'density', 'avg_degree', 
            'avg_clustering', 'diameter', 'avg_path_length',
            'connected_components'
        ],
        'centrality': [
            'degree_centrality', 'betweenness_centrality', 
            'closeness_centrality', 'eigenvector_centrality'
        ],
        'community': [
            'louvain', 'leiden', 'girvan_newman', 'label_propagation'
        ],
        'resilience': [
            'node_connectivity', 'edge_connectivity', 
            'average_node_connectivity'
        ],
        'spatial': [
            'beta_index', 'gamma_index', 'alpha_index',
            'cyclomatic_number', 'eta_index', 'pi_index'
        ]
    },
    
    # Configurações para grafos multi-camada
    'multi_layer': {
        'layers': ['erbs', 'edificacoes', 'vias', 'uso_solo'],
        'interlayer_connections': {
            'erbs-edificacoes': {
                'method': 'distance',
                'max_distance': 0.5  # em km
            },
            'erbs-vias': {
                'method': 'distance',
                'max_distance': 0.2  # em km 
            },
            'edificacoes-vias': {
                'method': 'nearest',
                'k': 3
            },
            'edificacoes-uso_solo': {
                'method': 'contains'
            }
        },
        'metrics': [
            'multi_degree', 'multi_betweenness', 'multi_closeness',
            'layer_interdependence', 'interlayer_correlation'
        ]
    },
    
    # Configurações para exportação para PyTorch Geometric
    'pyg_export': {
        'export_format': 'hetero_data',  # data, hetero_data, dataset
        'save_node_mapping': True,
        'node_features': True,
        'edge_features': True,
        'export_directory': 'data/pyg_models'
    }
}

# Configurações para o pipeline integrado
PIPELINE_CONFIG = {
    'steps': [
        {'name': 'load_data', 'enabled': True, 'params': {
            'data_directory': 'data/raw',
            'file_format': 'gpkg'
        }},
        {'name': 'preprocess_layers', 'enabled': True, 'params': LAYER_PREPROCESS_CONFIG},
        {'name': 'analyze_erb_coverage', 'enabled': True, 'params': ERB_COVERAGE_CONFIG},
        {'name': 'create_graphs', 'enabled': True, 'params': GRAPH_CONFIG},
        {'name': 'analyze_graphs', 'enabled': True, 'params': None},
        {'name': 'visualize_results', 'enabled': True, 'params': {
            'output_directory': 'output/visualizations',
            'interactive': True,
            'static': True,
            'formats': ['png', 'html', 'pdf']
        }},
        {'name': 'export_results', 'enabled': True, 'params': {
            'formats': ['gpkg', 'shp', 'csv', 'json'],
            'output_directory': 'output/results'
        }}
    ],
    
    # Configurações gerais
    'parallel': True,
    'num_workers': 4,
    'logger': {
        'level': 'INFO',
        'file': 'logs/pipeline.log',
        'console': True
    },
    'cache': {
        'enabled': True,
        'directory': 'cache',
        'max_size_gb': 2
    }
}

# Classes de erros personalizadas para o pipeline
class PipelineConfigError(Exception):
    """Erro na configuração do pipeline."""
    pass

class LayerPreprocessError(Exception):
    """Erro no pré-processamento de camadas."""
    pass

class ERBCoverageError(Exception):
    """Erro na análise de cobertura de ERBs."""
    pass

class GraphAnalysisError(Exception):
    """Erro na análise de grafos."""
    pass 