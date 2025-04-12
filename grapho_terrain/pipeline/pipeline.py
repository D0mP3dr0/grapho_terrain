"""
Pipeline integrado para processamento, análise e visualização de dados geoespaciais.

Este módulo implementa uma classe de pipeline que integra todos os componentes do projeto:
1. Pré-processamento de camadas geoespaciais (edifícios, vias, etc.)
2. Análise de cobertura de ERBs (Estações Rádio Base)
3. Criação e análise de grafos (redes) geoespaciais
4. Cálculo de métricas avançadas
5. Visualização de resultados

O pipeline permite executar os passos sequencialmente ou de forma independente,
mantendo um estado consistente entre as etapas.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Callable
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon

from ..network.base import GeoGraph
from ..network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
from ..telecommunications.erb import ERB, load_erb_data
from .preprocess import LayerProcessor, ERBProcessor
from .graph_builder import GraphBuilder
from .metrics import GraphMetrics, SpatialMetrics, MultiLayerMetrics


class PipelineStep:
    """Classe que representa um passo do pipeline."""
    
    def __init__(self, name: str, function: Callable, enabled: bool = True, params: Dict = None):
        """
        Inicializa um novo passo do pipeline.
        
        Args:
            name: Nome do passo
            function: Função a ser executada
            enabled: Se o passo está habilitado
            params: Parâmetros para a função
        """
        self.name = name
        self.function = function
        self.enabled = enabled
        self.params = params or {}
        self.result = None
        self.execution_time = 0
        self.status = "pending"
        self.error = None
    
    def execute(self, pipeline_context: Dict) -> Any:
        """
        Executa o passo do pipeline.
        
        Args:
            pipeline_context: Contexto do pipeline
            
        Returns:
            Resultado da execução
        """
        if not self.enabled:
            self.status = "skipped"
            return None
        
        try:
            self.status = "running"
            start_time = time.time()
            
            # Executa a função do passo com os parâmetros e contexto
            self.result = self.function(pipeline_context, **self.params)
            
            self.execution_time = time.time() - start_time
            self.status = "completed"
            return self.result
            
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logging.error(f"Erro no passo '{self.name}': {e}")
            raise e


class PipelineConfig:
    """Configuração do pipeline."""
    
    def __init__(self, config_dict: Dict = None, config_file: str = None):
        """
        Inicializa uma nova configuração de pipeline.
        
        Args:
            config_dict: Dicionário com configurações
            config_file: Caminho para arquivo JSON com configurações
        """
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
                
        if config_dict:
            self.config.update(config_dict)
    
    def get_step_config(self, step_name: str) -> Dict:
        """
        Obtém a configuração para um passo específico.
        
        Args:
            step_name: Nome do passo
            
        Returns:
            Configuração do passo
        """
        if 'steps' in self.config:
            for step in self.config['steps']:
                if step['name'] == step_name:
                    return step.get('params', {})
        
        return {}
    
    def is_step_enabled(self, step_name: str) -> bool:
        """
        Verifica se um passo está habilitado.
        
        Args:
            step_name: Nome do passo
            
        Returns:
            True se o passo está habilitado, False caso contrário
        """
        if 'steps' in self.config:
            for step in self.config['steps']:
                if step['name'] == step_name:
                    return step.get('enabled', True)
        
        return True
    
    def get_global_config(self) -> Dict:
        """
        Obtém a configuração global do pipeline.
        
        Returns:
            Configuração global
        """
        config = self.config.copy()
        if 'steps' in config:
            del config['steps']
        return config


class Pipeline:
    """Pipeline integrado para processamento, análise e visualização de dados geoespaciais."""
    
    def __init__(self, config: Union[Dict, PipelineConfig, str] = None):
        """
        Inicializa um novo pipeline.
        
        Args:
            config: Configuração do pipeline (dicionário, objeto PipelineConfig ou caminho para arquivo)
        """
        # Inicializa componentes
        self.layer_processor = LayerProcessor()
        self.graph_builder = GraphBuilder()
        
        # Inicializa contexto
        self.context = {
            'layers': {},          # Camadas geoespaciais carregadas
            'processed_layers': {},  # Camadas processadas
            'erbs': [],            # Lista de ERBs
            'erb_coverage': None,  # Camada de cobertura de ERBs
            'graphs': {},          # Grafos criados
            'metrics': {},         # Métricas calculadas
            'results': {},         # Resultados do pipeline
            'output': {}           # Saídas e visualizações
        }
        
        # Inicializa logger
        self.logger = self._setup_logger()
        
        # Configura o pipeline
        if isinstance(config, dict):
            self.config = PipelineConfig(config_dict=config)
        elif isinstance(config, PipelineConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = PipelineConfig(config_file=config)
        else:
            self.config = PipelineConfig()
        
        # Inicializa passos do pipeline
        self.steps = []
        self._setup_steps()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configura o logger do pipeline.
        
        Returns:
            Logger configurado
        """
        logger = logging.getLogger('grapho_terrain.pipeline')
        logger.setLevel(logging.INFO)
        
        # Cria handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Cria formato para logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Adiciona handler ao logger
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_steps(self):
        """Configura os passos do pipeline."""
        # Define passos padrão
        self.steps = [
            PipelineStep('load_data', self._load_data, 
                        enabled=self.config.is_step_enabled('load_data'),
                        params=self.config.get_step_config('load_data')),
            
            PipelineStep('preprocess_layers', self._preprocess_layers,
                        enabled=self.config.is_step_enabled('preprocess_layers'),
                        params=self.config.get_step_config('preprocess_layers')),
            
            PipelineStep('analyze_erb_coverage', self._analyze_erb_coverage,
                        enabled=self.config.is_step_enabled('analyze_erb_coverage'),
                        params=self.config.get_step_config('analyze_erb_coverage')),
            
            PipelineStep('create_graphs', self._create_graphs,
                        enabled=self.config.is_step_enabled('create_graphs'),
                        params=self.config.get_step_config('create_graphs')),
            
            PipelineStep('analyze_graphs', self._analyze_graphs,
                        enabled=self.config.is_step_enabled('analyze_graphs'),
                        params=self.config.get_step_config('analyze_graphs')),
            
            PipelineStep('visualize_results', self._visualize_results,
                        enabled=self.config.is_step_enabled('visualize_results'),
                        params=self.config.get_step_config('visualize_results')),
            
            PipelineStep('export_results', self._export_results,
                        enabled=self.config.is_step_enabled('export_results'),
                        params=self.config.get_step_config('export_results'))
        ]
    
    def run(self) -> Dict:
        """
        Executa todos os passos do pipeline.
        
        Returns:
            Contexto do pipeline com resultados
        """
        self.logger.info("Iniciando pipeline")
        start_time = time.time()
        
        for step in self.steps:
            if step.enabled:
                self.logger.info(f"Executando passo: {step.name}")
                try:
                    step.execute(self.context)
                    self.logger.info(f"Passo {step.name} concluído em {step.execution_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"Erro no passo {step.name}: {e}")
                    if self.config.config.get('stop_on_error', True):
                        break
            else:
                self.logger.info(f"Passo {step.name} desabilitado")
        
        total_time = time.time() - start_time
        self.logger.info(f"Pipeline concluído em {total_time:.2f}s")
        
        return self.context
    
    def run_step(self, step_name: str) -> Any:
        """
        Executa apenas um passo específico do pipeline.
        
        Args:
            step_name: Nome do passo a ser executado
            
        Returns:
            Resultado da execução do passo
        """
        for step in self.steps:
            if step.name == step_name and step.enabled:
                self.logger.info(f"Executando passo: {step.name}")
                result = step.execute(self.context)
                self.logger.info(f"Passo {step.name} concluído em {step.execution_time:.2f}s")
                return result
        
        self.logger.warning(f"Passo {step_name} não encontrado ou desabilitado")
        return None
    
    def run_steps(self, step_names: List[str]) -> Dict:
        """
        Executa uma sequência específica de passos do pipeline.
        
        Args:
            step_names: Lista de nomes dos passos a serem executados
            
        Returns:
            Contexto do pipeline com resultados
        """
        self.logger.info(f"Iniciando execução de passos: {', '.join(step_names)}")
        start_time = time.time()
        
        for step_name in step_names:
            self.run_step(step_name)
        
        total_time = time.time() - start_time
        self.logger.info(f"Execução concluída em {total_time:.2f}s")
        
        return self.context
    
    def _load_data(self, context: Dict, data_directory: str = 'data/raw', 
                  file_format: str = 'gpkg', layers: List[str] = None) -> Dict:
        """
        Carrega dados de arquivos para o pipeline.
        
        Args:
            context: Contexto do pipeline
            data_directory: Diretório com dados
            file_format: Formato dos arquivos
            layers: Lista de camadas a carregar
            
        Returns:
            Dicionário com camadas carregadas
        """
        loaded_layers = {}
        
        # Verifica se o diretório existe
        if not os.path.exists(data_directory):
            self.logger.warning(f"Diretório {data_directory} não encontrado")
            return loaded_layers
        
        # Lista arquivos do diretório
        files = [f for f in os.listdir(data_directory) if f.endswith(f'.{file_format}')]
        
        for file in files:
            layer_name = os.path.splitext(file)[0]
            
            # Filtra camadas se necessário
            if layers and layer_name not in layers:
                continue
            
            file_path = os.path.join(data_directory, file)
            
            try:
                # Carrega arquivo como GeoDataFrame
                gdf = gpd.read_file(file_path)
                loaded_layers[layer_name] = gdf
                self.logger.info(f"Camada {layer_name} carregada: {len(gdf)} registros")
            except Exception as e:
                self.logger.error(f"Erro ao carregar {file_path}: {e}")
        
        # Verifica se existem ERBs para carregar
        erb_file = os.path.join(data_directory, 'erbs.csv')
        if os.path.exists(erb_file):
            try:
                erbs = load_erb_data(erb_file)
                context['erbs'] = erbs
                self.logger.info(f"Dados de ERBs carregados: {len(erbs)} registros")
            except Exception as e:
                self.logger.error(f"Erro ao carregar ERBs: {e}")
        
        context['layers'] = loaded_layers
        return loaded_layers
    
    def _preprocess_layers(self, context: Dict, layers: Dict = None) -> Dict:
        """
        Pré-processa as camadas geoespaciais.
        
        Args:
            context: Contexto do pipeline
            layers: Configurações para processamento de camadas
            
        Returns:
            Dicionário com camadas processadas
        """
        processed_layers = {}
        
        # Usa camadas do contexto se não fornecidas
        input_layers = context['layers']
        if not input_layers:
            self.logger.warning("Nenhuma camada para processar")
            return processed_layers
        
        # Configurações de processamento
        layer_configs = layers.get('layers', {}) if layers else {}
        
        # Processa cada camada
        for layer_name, layer_data in input_layers.items():
            # Verifica se há configuração específica para a camada
            if layer_name in layer_configs:
                config = layer_configs[layer_name]
                source_type = config.get('source', layer_name)
                
                try:
                    # Processa a camada com base no tipo
                    processed = self.layer_processor.process_layer(
                        layer_data, source_type, **config)
                    
                    processed_layers[layer_name] = processed
                    self.logger.info(f"Camada {layer_name} processada: {len(processed)} registros")
                except Exception as e:
                    self.logger.error(f"Erro ao processar camada {layer_name}: {e}")
            else:
                # Mantém a camada sem processamento
                processed_layers[layer_name] = layer_data.copy()
                self.logger.info(f"Camada {layer_name} mantida sem processamento")
        
        context['processed_layers'] = processed_layers
        return processed_layers
    
    def _analyze_erb_coverage(self, context: Dict, **params) -> gpd.GeoDataFrame:
        """
        Analisa cobertura de ERBs.
        
        Args:
            context: Contexto do pipeline
            params: Parâmetros para análise de cobertura
            
        Returns:
            GeoDataFrame com cobertura de ERBs
        """
        erbs = context.get('erbs', [])
        if not erbs:
            self.logger.warning("Nenhuma ERB disponível para análise de cobertura")
            return None
        
        try:
            # Processa ERBs para calcular cobertura
            erb_processor = ERBProcessor()
            
            # Calcula raio de cobertura para as ERBs
            frequency = params.get('frequency', 1800)  # MHz
            power = params.get('power', 40)  # W
            gain = params.get('gain', 18)  # dBi
            height = params.get('height', 30)  # m
            
            for erb in erbs:
                if not hasattr(erb, 'frequency') or not erb.frequency:
                    erb.frequency = frequency
                if not hasattr(erb, 'power') or not erb.power:
                    erb.power = power
                if not hasattr(erb, 'gain') or not erb.gain:
                    erb.gain = gain
                if not hasattr(erb, 'height') or not erb.height:
                    erb.height = height
                
                # Calcula EIRP e raio de cobertura
                erb_processor.calculate_coverage(erb)
            
            # Cria camada de cobertura
            coverage_params = {
                'create_buffer': params.get('create_buffer', True),
                'resolution': params.get('resolution', 50)
            }
            coverage = erb_processor.create_coverage_layer(erbs, **coverage_params)
            
            context['erb_coverage'] = coverage
            self.logger.info(f"Cobertura de ERBs calculada: {len(coverage)} setores")
            
            return coverage
            
        except Exception as e:
            self.logger.error(f"Erro na análise de cobertura de ERBs: {e}")
            return None
    
    def _create_graphs(self, context: Dict, **params) -> Dict:
        """
        Cria grafos a partir das camadas processadas.
        
        Args:
            context: Contexto do pipeline
            params: Parâmetros para criação de grafos
            
        Returns:
            Dicionário com grafos criados
        """
        processed_layers = context.get('processed_layers', {})
        erbs = context.get('erbs', [])
        
        if not processed_layers and not erbs:
            self.logger.warning("Nenhuma camada processada ou ERB disponível para criar grafos")
            return {}
        
        # Configurações para grafos
        graph_types = params.get('graph_types', [])
        multilayer = params.get('multilayer', False)
        
        graphs = {}
        
        # Cria grafos individuais para cada camada conforme solicitado
        for graph_type in graph_types:
            try:
                if graph_type == 'roads' and 'vias' in processed_layers:
                    roads_params = params.get('roads_params', {})
                    road_graph = self.graph_builder.create_road_graph(
                        processed_layers['vias'],
                        directed=roads_params.get('directed', False),
                        weight_field=roads_params.get('weight_field', None),
                        simplify=roads_params.get('simplify', True)
                    )
                    graphs['road_graph'] = road_graph
                    self.logger.info(f"Grafo de vias criado: {road_graph.number_of_nodes()} nós, {road_graph.number_of_edges()} arestas")
                
                elif graph_type == 'buildings' and 'edificacoes' in processed_layers:
                    buildings_params = params.get('buildings_params', {})
                    building_graph = self.graph_builder.create_building_graph(
                        processed_layers['edificacoes'],
                        connectivity_threshold=buildings_params.get('connectivity_threshold', 50.0),
                        weight_field=buildings_params.get('weight_field', None)
                    )
                    graphs['building_graph'] = building_graph
                    self.logger.info(f"Grafo de edifícios criado: {building_graph.number_of_nodes()} nós, {building_graph.number_of_edges()} arestas")
                
                elif graph_type == 'erb' and erbs:
                    erb_params = params.get('erb_params', {})
                    erb_graph = self.graph_builder.create_erb_network(
                        erbs,
                        connectivity_type=erb_params.get('connectivity_type', 'distance'),
                        max_distance=erb_params.get('max_distance', 5000.0),
                        k_nearest=erb_params.get('k_nearest', 3)
                    )
                    graphs['erb_graph'] = erb_graph
                    self.logger.info(f"Grafo de ERBs criado: {erb_graph.number_of_nodes()} nós, {erb_graph.number_of_edges()} arestas")
                
                # Feature graphs para uso em ML
                elif graph_type == 'feature_graph' and processed_layers:
                    for layer_name, layer in processed_layers.items():
                        feature_params = params.get('feature_params', {}).get(layer_name, {})
                        if feature_params:
                            feature_graph = self.graph_builder.create_feature_graph(
                                layer,
                                node_features=feature_params.get('node_features', None),
                                edge_features=feature_params.get('edge_features', None),
                                connectivity_type=feature_params.get('connectivity_type', 'knn'),
                                k_nearest=feature_params.get('k_nearest', 3),
                                max_distance=feature_params.get('max_distance', None)
                            )
                            graphs[f'{layer_name}_feature_graph'] = feature_graph
                            self.logger.info(f"Grafo de características para {layer_name} criado: {feature_graph.number_of_nodes()} nós, {feature_graph.number_of_edges()} arestas")
                
            except Exception as e:
                self.logger.error(f"Erro ao criar grafo {graph_type}: {e}")
        
        # Cria grafo multicamadas se solicitado
        if multilayer:
            try:
                multilayer_params = params.get('multilayer_params', {})
                
                # Prepara camadas para o grafo multicamadas
                layers_dict = {}
                for layer_name, layer in processed_layers.items():
                    if layer_name in multilayer_params.get('layers', []):
                        layers_dict[layer_name] = layer
                
                # Configura conectores entre camadas
                layer_connectors = multilayer_params.get('layer_connectors', {})
                connectors_dict = {}
                
                for connection, config in layer_connectors.items():
                    layer1, layer2 = connection.split('-')
                    connectors_dict[(layer1, layer2)] = config
                
                # Cria o grafo multicamadas
                if layers_dict:
                    multi_graph = self.graph_builder.create_multiplex_graph(
                        layers_dict, connectors_dict
                    )
                    graphs['multilayer_graph'] = multi_graph
                    self.logger.info(f"Grafo multicamadas criado com {len(multi_graph.layers)} camadas")
            
            except Exception as e:
                self.logger.error(f"Erro ao criar grafo multicamadas: {e}")
        
        context['graphs'] = graphs
        return graphs
    
    def _analyze_graphs(self, context: Dict, **params) -> Dict:
        """
        Analisa os grafos calculando métricas.
        
        Args:
            context: Contexto do pipeline
            params: Parâmetros para análise de grafos
            
        Returns:
            Dicionário com métricas calculadas
        """
        graphs = context.get('graphs', {})
        if not graphs:
            self.logger.warning("Nenhum grafo disponível para análise")
            return {}
        
        metrics_results = {}
        
        # Métricas a calcular
        metrics_to_calculate = params.get('metrics', {
            'basic': True,
            'centrality': True,
            'community': False,
            'resilience': False
        })
        
        # Calcula métricas para cada grafo
        for graph_name, graph in graphs.items():
            try:
                # Pula grafo multicamadas (tratado separadamente)
                if graph_name == 'multilayer_graph':
                    continue
                
                # Instancia calculador de métricas
                metrics_calculator = GraphMetrics(graph)
                graph_metrics = {}
                
                # Calcula métricas básicas
                if metrics_to_calculate.get('basic', True):
                    basic = metrics_calculator.basic_metrics()
                    graph_metrics.update(basic)
                
                # Calcula métricas de centralidade
                if metrics_to_calculate.get('centrality', True):
                    centrality = metrics_calculator.centrality_metrics()
                    graph_metrics.update(centrality)
                
                # Calcula métricas de comunidade
                if metrics_to_calculate.get('community', False):
                    community = metrics_calculator.community_metrics()
                    graph_metrics.update(community)
                
                # Calcula métricas de resiliência
                if metrics_to_calculate.get('resilience', False):
                    resilience = metrics_calculator.resilience_metrics()
                    graph_metrics.update(resilience)
                
                metrics_results[graph_name] = graph_metrics
                self.logger.info(f"Métricas calculadas para {graph_name}: {len(graph_metrics)} métricas")
                
            except Exception as e:
                self.logger.error(f"Erro ao calcular métricas para {graph_name}: {e}")
        
        # Calcula métricas para grafo multicamadas
        if 'multilayer_graph' in graphs:
            try:
                multi_graph = graphs['multilayer_graph']
                multi_metrics = MultiLayerMetrics(multi_graph)
                
                # Obtém métricas para o grafo multicamadas
                # (métodos específicos seriam implementados na classe MultiLayerMetrics)
                metrics_results['multilayer_graph'] = multi_metrics.layer_metrics
                self.logger.info(f"Métricas calculadas para grafo multicamadas")
                
            except Exception as e:
                self.logger.error(f"Erro ao calcular métricas para grafo multicamadas: {e}")
        
        context['metrics'] = metrics_results
        return metrics_results
    
    def _visualize_results(self, context: Dict, **params) -> Dict:
        """
        Cria visualizações dos resultados.
        
        Args:
            context: Contexto do pipeline
            params: Parâmetros para visualização
            
        Returns:
            Dicionário com visualizações
        """
        # Parâmetros para visualização
        output_directory = params.get('output_directory', 'output/visualizations')
        interactive = params.get('interactive', True)
        static = params.get('static', True)
        formats = params.get('formats', ['png', 'html'])
        
        # Cria diretório de saída se não existir
        os.makedirs(output_directory, exist_ok=True)
        
        visualizations = {}
        
        # Visualiza camadas processadas
        processed_layers = context.get('processed_layers', {})
        for layer_name, layer in processed_layers.items():
            try:
                # Visualização estática
                if static:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    layer.plot(ax=ax)
                    ax.set_title(f'Camada: {layer_name}')
                    
                    # Salva em cada formato solicitado
                    for fmt in formats:
                        if fmt != 'html':  # html é apenas para interativo
                            fig_path = os.path.join(output_directory, f'layer_{layer_name}.{fmt}')
                            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    
                    plt.close(fig)
                    
                    visualizations[f'layer_{layer_name}'] = 'static_map'
                
                # Visualização interativa
                if interactive:
                    # A visualização interativa seria implementada com folium
                    # Código para criar mapa interativo
                    pass
                    
            except Exception as e:
                self.logger.error(f"Erro ao visualizar camada {layer_name}: {e}")
        
        # Visualiza grafos
        graphs = context.get('graphs', {})
        for graph_name, graph in graphs.items():
            try:
                # Visualização estática do grafo
                if static:
                    fig, ax = plt.subplots(figsize=(12, 12))
                    
                    # Posições dos nós
                    pos = nx.get_node_attributes(graph, 'pos')
                    if not pos:
                        # Se não tiver posições definidas, usa layout spring
                        pos = nx.spring_layout(graph)
                    
                    # Desenha o grafo
                    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
                    nx.draw_networkx_nodes(graph, pos, node_size=20, alpha=0.5, ax=ax)
                    
                    ax.set_title(f'Grafo: {graph_name}')
                    ax.set_axis_off()
                    
                    # Salva em cada formato solicitado
                    for fmt in formats:
                        if fmt != 'html':  # html é apenas para interativo
                            fig_path = os.path.join(output_directory, f'graph_{graph_name}.{fmt}')
                            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    
                    plt.close(fig)
                    
                    visualizations[f'graph_{graph_name}'] = 'static_graph'
                
                # Visualização interativa do grafo
                if interactive:
                    # Código para criar visualização interativa
                    pass
                    
            except Exception as e:
                self.logger.error(f"Erro ao visualizar grafo {graph_name}: {e}")
        
        # Visualiza métricas
        metrics = context.get('metrics', {})
        for graph_name, graph_metrics in metrics.items():
            try:
                # Exporta métricas para CSV
                metrics_df = pd.DataFrame(graph_metrics)
                csv_path = os.path.join(output_directory, f'metrics_{graph_name}.csv')
                metrics_df.to_csv(csv_path)
                
                visualizations[f'metrics_{graph_name}'] = 'csv'
                
                # Cria gráficos para métricas principais
                # (distribuição de grau, centralidade, etc.)
                # Código para gráficos de métricas
                
            except Exception as e:
                self.logger.error(f"Erro ao visualizar métricas de {graph_name}: {e}")
        
        context['visualizations'] = visualizations
        return visualizations
    
    def _export_results(self, context: Dict, **params) -> Dict:
        """
        Exporta os resultados do pipeline para arquivos.
        
        Args:
            context: Contexto do pipeline
            params: Parâmetros para exportação
            
        Returns:
            Dicionário com resultados exportados
        """
        # Parâmetros para exportação
        output_directory = params.get('output_directory', 'output/results')
        formats = params.get('formats', ['gpkg', 'shp', 'csv', 'json'])
        
        # Cria diretório de saída se não existir
        os.makedirs(output_directory, exist_ok=True)
        
        exports = {}
        
        # Exporta camadas processadas
        processed_layers = context.get('processed_layers', {})
        for layer_name, layer in processed_layers.items():
            try:
                for fmt in formats:
                    if fmt in ['gpkg', 'shp']:
                        # Exporta camada para formato geoespacial
                        file_path = os.path.join(output_directory, f'{layer_name}.{fmt}')
                        layer.to_file(file_path, driver='GPKG' if fmt == 'gpkg' else 'ESRI Shapefile')
                        exports[f'layer_{layer_name}_{fmt}'] = file_path
                    
                    elif fmt == 'csv':
                        # Exporta atributos para CSV
                        file_path = os.path.join(output_directory, f'{layer_name}.csv')
                        
                        # Cria colunas x, y para geometria
                        data = layer.copy()
                        if 'geometry' in data.columns:
                            if all(isinstance(geom, Point) for geom in data.geometry):
                                data['x'] = data.geometry.x
                                data['y'] = data.geometry.y
                            elif all(isinstance(geom, Polygon) for geom in data.geometry):
                                centroids = data.geometry.centroid
                                data['x'] = centroids.x
                                data['y'] = centroids.y
                        
                        # Remove coluna de geometria para CSV
                        if 'geometry' in data.columns:
                            data = data.drop(columns=['geometry'])
                            
                        data.to_csv(file_path, index=False)
                        exports[f'layer_{layer_name}_csv'] = file_path
                    
                    elif fmt == 'json':
                        # Exporta para GeoJSON
                        file_path = os.path.join(output_directory, f'{layer_name}.geojson')
                        layer.to_file(file_path, driver='GeoJSON')
                        exports[f'layer_{layer_name}_json'] = file_path
                        
            except Exception as e:
                self.logger.error(f"Erro ao exportar camada {layer_name}: {e}")
        
        # Exporta resultados de métricas
        metrics = context.get('metrics', {})
        for graph_name, graph_metrics in metrics.items():
            try:
                # Exporta para CSV
                file_path = os.path.join(output_directory, f'metrics_{graph_name}.csv')
                metrics_df = pd.DataFrame({k: [v] for k, v in graph_metrics.items()})
                metrics_df.to_csv(file_path, index=False)
                exports[f'metrics_{graph_name}_csv'] = file_path
                
                # Exporta para JSON
                file_path = os.path.join(output_directory, f'metrics_{graph_name}.json')
                with open(file_path, 'w') as f:
                    json.dump(graph_metrics, f, indent=2)
                exports[f'metrics_{graph_name}_json'] = file_path
                
            except Exception as e:
                self.logger.error(f"Erro ao exportar métricas de {graph_name}: {e}")
        
        # Exporta grafos
        graphs = context.get('graphs', {})
        for graph_name, graph in graphs.items():
            try:
                # Exporta para GraphML
                file_path = os.path.join(output_directory, f'{graph_name}.graphml')
                nx.write_graphml(graph, file_path)
                exports[f'graph_{graph_name}_graphml'] = file_path
                
                # Exporta para pickle de NetworkX
                file_path = os.path.join(output_directory, f'{graph_name}.pickle')
                with open(file_path, 'wb') as f:
                    pickle.dump(graph, f)
                exports[f'graph_{graph_name}_pickle'] = file_path
                
            except Exception as e:
                self.logger.error(f"Erro ao exportar grafo {graph_name}: {e}")
        
        context['exports'] = exports
        return exports
    
    def save_context(self, file_path: str) -> None:
        """
        Salva o contexto atual do pipeline para um arquivo.
        
        Args:
            file_path: Caminho para o arquivo
        """
        # Cria um dicionário com o estado do pipeline
        state = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.config,
            'steps': [
                {
                    'name': step.name,
                    'enabled': step.enabled,
                    'status': step.status,
                    'execution_time': step.execution_time,
                    'error': step.error
                }
                for step in self.steps
            ],
            'context': {
                # Remove objetos não serializáveis
                'layers': {k: f"GeoDataFrame({len(v)} rows)" for k, v in self.context.get('layers', {}).items()},
                'processed_layers': {k: f"GeoDataFrame({len(v)} rows)" for k, v in self.context.get('processed_layers', {}).items()},
                'erbs': f"List[ERB]({len(self.context.get('erbs', []))} items)",
                'erb_coverage': f"GeoDataFrame({len(self.context.get('erb_coverage', [])) if self.context.get('erb_coverage') is not None else 0} rows)",
                'graphs': {k: f"Graph({v.number_of_nodes()} nodes, {v.number_of_edges()} edges)" for k, v in self.context.get('graphs', {}).items()},
                'metrics': self.context.get('metrics', {}),
                'results': self.context.get('results', {}),
                'output': self.context.get('output', {})
            }
        }
        
        # Salva em JSON
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        self.logger.info(f"Contexto do pipeline salvo em {file_path}")
    
    def load_context(self, file_path: str) -> bool:
        """
        Carrega o contexto de um arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            True se o contexto foi carregado com sucesso, False caso contrário
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Arquivo {file_path} não encontrado")
            return False
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Carrega configuração
            self.config = PipelineConfig(config_dict=state.get('config', {}))
            
            # Reconfigura passos
            self._setup_steps()
            
            # Atualiza status dos passos
            for saved_step in state.get('steps', []):
                for step in self.steps:
                    if step.name == saved_step['name']:
                        step.enabled = saved_step.get('enabled', True)
                        step.status = saved_step.get('status', 'pending')
                        step.execution_time = saved_step.get('execution_time', 0)
                        step.error = saved_step.get('error', None)
            
            self.logger.info(f"Contexto do pipeline carregado de {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar contexto: {e}")
            return False


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("="*30 + " INICIANDO TESTE DO PIPELINE " + "="*30)

    # Configuração padrão (pode ser personalizada aqui ou via arquivo JSON)
    # Usará 'data/raw' como diretório de dados padrão
    # e tentará carregar arquivos .gpkg de lá
    config = {
        "stop_on_error": False, # Continuar mesmo se um passo falhar
        "steps": {
            # Exemplo de como desabilitar um passo:
            # { "name": "analyze_erb_coverage", "enabled": False },
            # Exemplo de como passar parâmetros para um passo:
            # { "name": "create_graphs", "params": { "multilayer": True, "graph_types": ["roads"] } }
        },
        # Adicione outras configurações globais aqui se necessário
        "global_param_exemplo": "valor_exemplo"
    }

    # Instanciar e executar o pipeline
    try:
        pipeline_instance = Pipeline(config=config)
        pipeline_context = pipeline_instance.run()

        logger.info("="*30 + " TESTE DO PIPELINE CONCLUÍDO " + "="*30)

        # Opcional: inspecionar o contexto final
        # logger.info("\nContexto final do pipeline:")
        # for key, value in pipeline_context.items():
        #     if isinstance(value, (dict, list)):
        #         logger.info(f" - {key}: (Tipo: {type(value).__name__}, Itens: {len(value)})")
        #     else:
        #          logger.info(f" - {key}: {value}")

    except Exception as main_error:
        logger.error(f"Erro fatal durante a execução do pipeline: {main_error}")
        import traceback
        logger.error(traceback.format_exc()) 