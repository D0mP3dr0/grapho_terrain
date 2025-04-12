#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline completo para Grapho Terrain

Este script implementa um fluxo contínuo e integrado de todo o processo do Grapho Terrain,
garantindo que a execução ocorra desde o carregamento de dados até a criação do grafo final,
incluindo a geração de todas as visualizações e análises em um único fluxo.

Fluxo de execução:
1. Carregamento dos dados de entrada (ERBs, edificações, vias, etc.)
2. Pré-processamento das camadas geoespaciais
3. Análise de cobertura de ERBs
4. Criação de grafos de camada única
5. Criação de grafo multicamada
6. Análise de métricas de rede
7. Visualização dos resultados
8. Exportação dos dados processados

Uso:
    python examples/pipeline_completo.py [--config CONFIG_FILE] [--data DATA_DIR] [--output OUTPUT_DIR]
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Importações do projeto
from grapho_terrain.pipeline.pipeline import Pipeline, PipelineConfig
from grapho_terrain.telecommunications.erb import carregar_dados_erbs_personalizado
from grapho_terrain.pipeline.preprocess import LayerProcessor, ERBProcessor
from grapho_terrain.network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
from grapho_terrain.pipeline.visualizer import Visualizer
from grapho_terrain.pipeline.graph_builder import GraphBuilder
from grapho_terrain.pipeline.metrics import GraphMetrics

# Configuração do logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pipeline_completo')


def parse_arguments():
    """Processa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Pipeline completo para Grapho Terrain')
    
    parser.add_argument('--config', type=str, 
                       help='Caminho para arquivo de configuração JSON (opcional)')
    
    parser.add_argument('--data', type=str, default='data',
                       help='Diretório com dados de entrada (padrão: data)')
    
    parser.add_argument('--output', type=str, default='output',
                       help='Diretório para resultados (padrão: output)')
    
    parser.add_argument('--sorocaba', action='store_true',
                       help='Usar exemplo de Sorocaba com ERBs')
    
    parser.add_argument('--synthetic', action='store_true',
                       help='Gerar dados sintéticos para teste')
    
    return parser.parse_args()


def gerar_dados_sinteticos(output_dir):
    """
    Gera dados sintéticos para teste do pipeline.
    
    Args:
        output_dir: Diretório onde os dados serão salvos
    """
    logger.info("Gerando dados sintéticos para teste...")
    
    # Cria diretório de saída
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Gera edificações
    buildings_data = []
    for i in range(100):
        # Coordenadas aleatórias
        x = np.random.uniform(-46.65, -46.60)
        y = np.random.uniform(-23.57, -23.54)
        
        # Polígono para o edifício
        width = np.random.uniform(0.0005, 0.002)
        height = np.random.uniform(0.0005, 0.002)
        from shapely.geometry import Polygon
        polygon = Polygon([
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ])
        
        # Atributos
        buildings_data.append({
            'id': f'B{i}',
            'altura': np.random.uniform(5, 100),
            'tipo': np.random.choice(['residencial', 'comercial', 'industrial']),
            'area': width * height * 111000 * 111000,
            'geometry': polygon
        })
    
    buildings_gdf = gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")
    buildings_gdf.to_file(os.path.join(raw_dir, 'buildings.gpkg'), driver='GPKG')
    
    # Gera vias
    roads_data = []
    for i in range(50):
        x1 = np.random.uniform(-46.65, -46.60)
        y1 = np.random.uniform(-23.57, -23.54)
        length = np.random.uniform(0.005, 0.02)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x2 = x1 + length * np.cos(angle)
        y2 = y1 + length * np.sin(angle)
        from shapely.geometry import LineString
        line = LineString([(x1, y1), (x2, y2)])
        
        roads_data.append({
            'id': f'R{i}',
            'tipo': np.random.choice(['primaria', 'secundaria', 'local']),
            'largura': np.random.uniform(5, 20),
            'nome': f'Via {i}',
            'geometry': line
        })
    
    roads_gdf = gpd.GeoDataFrame(roads_data, crs="EPSG:4326")
    roads_gdf.to_file(os.path.join(raw_dir, 'roads.gpkg'), driver='GPKG')
    
    # Gera ERBs
    erbs_data = []
    operadoras = ['Claro', 'Vivo', 'TIM', 'Oi']
    tecnologias = ['4G', '5G']
    frequencias = [700, 850, 1800, 2100, 2600, 3500]
    
    for i in range(20):
        x = np.random.uniform(-46.65, -46.60)
        y = np.random.uniform(-23.57, -23.54)
        from shapely.geometry import Point
        point = Point(x, y)
        
        operadora = np.random.choice(operadoras)
        tecnologia = np.random.choice(tecnologias)
        frequencia = np.random.choice(frequencias)
        
        erbs_data.append({
            'id': f'ERB{i:03d}',
            'nome': f'ERB-{operadora}-{i}',
            'operadora': operadora,
            'tecnologia': tecnologia,
            'freq_mhz': frequencia,
            'potencia_watts': np.random.uniform(10, 40),
            'ganho_antena': np.random.uniform(10, 18),
            'azimute': np.random.uniform(0, 360),
            'altura_m': np.random.uniform(15, 45),
            'longitude': x,
            'latitude': y,
            'geometry': point
        })
    
    erbs_gdf = gpd.GeoDataFrame(erbs_data, crs="EPSG:4326")
    erbs_gdf.to_file(os.path.join(raw_dir, 'erbs.gpkg'), driver='GPKG')
    erbs_df = pd.DataFrame(erbs_data)
    erbs_df.to_csv(os.path.join(raw_dir, 'erbs.csv'), index=False)
    
    logger.info(f"Dados sintéticos gerados em {raw_dir}")
    return raw_dir


def carregar_dados_sorocaba(data_dir):
    """
    Carrega dados do exemplo de Sorocaba.
    
    Args:
        data_dir: Diretório base para dados
    
    Returns:
        Dicionário com layers carregadas
    """
    logger.info("Carregando dados de ERBs de Sorocaba...")
    
    # Verifica se o arquivo existe
    arquivo_csv = os.path.join(data_dir, 'sorocaba_erbs.csv')
    if not os.path.exists(arquivo_csv):
        # Tentando encontrar em outra localização
        arquivo_csv = 'data/sorocaba_erbs.csv'
        if not os.path.exists(arquivo_csv):
            logger.warning(f"Arquivo de ERBs de Sorocaba não encontrado.")
            # Criar dados de exemplo
            from examples.sorocaba.erb_sorocaba import criar_dados_exemplo
            criar_dados_exemplo(arquivo_csv)
    
    # Carrega ERBs com sistema de coordenadas correto
    erbs = carregar_dados_erbs_personalizado(
        arquivo_csv,
        crs_origem="EPSG:31983",  # SIRGAS 2000 / UTM zone 23S (comum em Sorocaba)
        crs_destino="EPSG:4326",  # WGS84 (padrão global)
        lon_col='longitude',
        lat_col='latitude'
    )
    
    gdf_erbs = erbs.to_geodataframe()
    
    # Criar setores de cobertura
    gdf_setores = erbs.create_coverage_sectors(tipo_area='urbana')
    
    # Retorna como dicionário de layers
    return {
        'erbs': gdf_erbs,
        'setores': gdf_setores
    }


def executar_pipeline(config=None, data_dir='data', output_dir='output', use_sorocaba=False):
    """
    Executa o pipeline completo do Grapho Terrain.
    
    Args:
        config: Configuração do pipeline (arquivo ou dicionário)
        data_dir: Diretório com dados de entrada
        output_dir: Diretório para resultados
        use_sorocaba: Se deve usar o exemplo específico de Sorocaba
    """
    start_time = time.time()
    logger.info("Iniciando pipeline completo...")
    
    # Cria diretórios de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # ETAPA 1: CARREGAMENTO DE DADOS
    logger.info("1. Carregamento de dados")
    
    if use_sorocaba:
        # Usa o exemplo específico de Sorocaba
        layers = carregar_dados_sorocaba(data_dir)
        
        # Salva visualizações básicas
        sorocaba_dir = os.path.join(output_dir, 'sorocaba')
        os.makedirs(sorocaba_dir, exist_ok=True)
        
        visualizer = Visualizer(figsize=(12, 10), basemap=True)
        
        # Visualização das ERBs
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        visualizer.plot_points(layers['erbs'], ax=ax1, color='red', size=50, 
                              alpha=0.7, label='ERBs')
        ax1.set_title("ERBs de Sorocaba")
        visualizer.save_figure(fig1, os.path.join(sorocaba_dir, "erbs_sorocaba.png"))
        
        # Visualização dos setores de cobertura
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        visualizer.plot_layer(layers['setores'], ax=ax2, column='operadora', 
                             alpha=0.5, legend=True)
        visualizer.plot_points(layers['erbs'], ax=ax2, color='black', size=20, 
                              alpha=1.0, label='ERBs')
        ax2.set_title("Cobertura de ERBs em Sorocaba")
        visualizer.save_figure(fig2, os.path.join(sorocaba_dir, "cobertura_erbs_sorocaba.png"))
        
    else:
        # Usa o pipeline padrão para carregar dados
        pipeline = Pipeline(config)
        pipeline.run_step('load_data')
        layers = pipeline.context['layers']
        
        if not layers:
            logger.warning("Nenhum dado encontrado. Verifique o diretório de dados.")
            return
    
    # ETAPA 2: PRÉ-PROCESSAMENTO
    logger.info("2. Pré-processamento de camadas")
    
    layer_processor = LayerProcessor()
    processed_layers = {}
    
    # Processa cada camada
    for name, layer in layers.items():
        if name == 'erbs':
            # Processamento especial para ERBs
            erb_processor = ERBProcessor()
            erbs = erb_processor.dataframe_to_erbs(layer)
            
            # Calcula métricas para cada ERB
            from grapho_terrain.telecommunications.erb import compute_coverage_radius, compute_eirp
            for erb in erbs:
                if not hasattr(erb, 'eirp') or erb.eirp is None:
                    erb.eirp = compute_eirp(erb.power, gain=17.0)
                
                if not hasattr(erb, 'coverage_radius') or erb.coverage_radius is None:
                    erb.coverage_radius = compute_coverage_radius(
                        erb.eirp, 
                        erb.frequency, 
                        sensitivity=-110.0,
                        terrain_factor=1.0
                    )
            
            processed_layers[name] = erb_processor.erbs_to_dataframe(erbs)
            
        elif name == 'buildings' or name == 'edificacoes':
            # Processamento para edificações
            processed_layers[name] = layer_processor.process_layer(
                layer, 
                layer_type='building',
                create_centroids=True
            )
            
        elif name == 'roads' or name == 'vias':
            # Processamento para vias
            processed_layers[name] = layer_processor.process_layer(
                layer, 
                layer_type='road',
                simplify_tolerance=0.0001
            )
            
        elif name == 'setores':
            # Setores de cobertura já processados
            processed_layers[name] = layer
            
        else:
            # Processamento genérico para outras camadas
            processed_layers[name] = layer_processor.process_layer(layer)
    
    # ETAPA 3: CRIAÇÃO DE GRAFOS
    logger.info("3. Criação de grafos")
    
    graph_builder = GraphBuilder()
    graphs = {}
    
    # Cria grafos para cada camada processada
    for name, layer in processed_layers.items():
        if name == 'erbs':
            # Grafo de ERBs
            erb_graph = FeatureGeoGraph()
            erb_graph.create_from_points(
                layer,
                id_col='id',
                feature_cols=['operadora', 'tecnologia', 'freq_mhz', 'potencia_watts',
                            'ganho_antena', 'altura_m', 'azimute'],
                encode_categorical=True
            )
            graphs[name] = erb_graph
            
        elif name == 'buildings' or name == 'edificacoes':
            # Grafo de edificações
            building_graph = FeatureGeoGraph()
            if 'centroid' in layer.columns:
                building_graph.create_from_points(
                    layer,
                    id_col='id',
                    feature_cols=['altura', 'tipo', 'area'],
                    encode_categorical=True,
                    point_geometry_col='centroid'
                )
            else:
                building_graph.create_from_polygons(
                    layer,
                    id_col='id',
                    feature_cols=['altura', 'tipo', 'area'],
                    encode_categorical=True,
                    use_centroids=True
                )
            graphs[name] = building_graph
            
        elif name == 'roads' or name == 'vias':
            # Grafo de vias
            road_graph = FeatureGeoGraph()
            road_graph.create_from_lines(
                layer,
                id_col='id',
                feature_cols=['tipo', 'largura', 'nome'],
                encode_categorical=True,
                split_lines=True
            )
            graphs[name] = road_graph
            
        elif name != 'setores':  # Ignora os setores para grafos
            # Grafo genérico para outras camadas
            generic_graph = FeatureGeoGraph()
            if layer.geometry.iloc[0].geom_type == 'Point':
                generic_graph.create_from_points(layer, id_col='id')
            elif layer.geometry.iloc[0].geom_type == 'LineString':
                generic_graph.create_from_lines(layer, id_col='id')
            elif layer.geometry.iloc[0].geom_type == 'Polygon':
                generic_graph.create_from_polygons(layer, id_col='id', use_centroids=True)
            
            graphs[name] = generic_graph
    
    # ETAPA 4: CRIAÇÃO DE GRAFO MULTICAMADA
    logger.info("4. Criação de grafo multicamada")
    
    # Verifica quais camadas estão disponíveis
    available_layers = list(graphs.keys())
    logger.info(f"Camadas disponíveis para grafo multicamada: {available_layers}")
    
    # Cria grafo multicamada com as camadas disponíveis
    multi_graph = MultiLayerFeatureGraph()
    
    # Adiciona as camadas disponíveis
    for name, graph in graphs.items():
        multi_graph.add_layer(name, graph)
    
    # Adiciona conexões entre camadas se tiver pelo menos duas camadas
    if len(available_layers) >= 2:
        # Se temos ERBs e edificações
        if 'erbs' in graphs and ('buildings' in graphs or 'edificacoes' in graphs):
            building_layer = 'buildings' if 'buildings' in graphs else 'edificacoes'
            multi_graph.add_interlayer_edges_knn('erbs', building_layer, k=3, max_distance=0.01)
            logger.info(f"Adicionada conexão entre ERBs e {building_layer}")
        
        # Se temos edificações e vias
        if ('buildings' in graphs or 'edificacoes' in graphs) and ('roads' in graphs or 'vias' in graphs):
            building_layer = 'buildings' if 'buildings' in graphs else 'edificacoes'
            road_layer = 'roads' if 'roads' in graphs else 'vias'
            multi_graph.add_interlayer_edges_knn(building_layer, road_layer, k=2, max_distance=0.01)
            logger.info(f"Adicionada conexão entre {building_layer} e {road_layer}")
    
    # ETAPA 5: ANÁLISE DE MÉTRICAS
    logger.info("5. Análise de métricas de rede")
    
    # Analisa métricas para cada grafo
    metrics = {}
    
    for name, graph in graphs.items():
        graph_metrics = GraphMetrics(graph.graph)
        
        # Métricas básicas
        basic_metrics = graph_metrics.calculate_basic_metrics()
        
        # Métricas de centralidade
        centrality_metrics = graph_metrics.calculate_centrality_metrics()
        
        # Combina as métricas
        metrics[name] = {
            'basic': basic_metrics,
            'centrality': centrality_metrics
        }
    
    # Métricas para o grafo multicamada
    if len(available_layers) >= 2:
        multi_metrics = {
            'num_layers': len(multi_graph.layers),
            'interlayer_edges': len(multi_graph.interlayer_edges),
            'total_nodes': sum(graph.graph.number_of_nodes() for graph in graphs.values()),
            'total_edges': sum(graph.graph.number_of_edges() for graph in graphs.values())
        }
        metrics['multi'] = multi_metrics
    
    # ETAPA 6: VISUALIZAÇÃO
    logger.info("6. Criação de visualizações")
    
    # Cria diretório para visualizações
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    visualizer = Visualizer(figsize=(12, 10), basemap=True)
    
    # Visualiza cada camada
    for name, layer in processed_layers.items():
        if name == 'setores':
            continue  # Já visualizamos os setores antes
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if name == 'erbs':
            visualizer.plot_erb_data(layer, ax=ax, plot_coverage=True, 
                                   by_operator=True, title=f"Camada de {name}")
        elif layer.geometry.iloc[0].geom_type == 'Point':
            visualizer.plot_points(layer, ax=ax, size=20, alpha=0.7, 
                                 title=f"Camada de {name}")
        elif layer.geometry.iloc[0].geom_type == 'LineString':
            visualizer.plot_layer(layer, ax=ax, color='blue', alpha=0.7, 
                                title=f"Camada de {name}")
        elif layer.geometry.iloc[0].geom_type == 'Polygon':
            visualizer.plot_layer(layer, ax=ax, alpha=0.5, 
                                title=f"Camada de {name}")
            
        visualizer.save_figure(fig, os.path.join(viz_dir, f"camada_{name}.png"))
    
    # Visualiza grafos individuais
    for name, graph in graphs.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Verifica se temos métricas de centralidade para colorir o grafo
        if name in metrics and 'centrality' in metrics[name]:
            node_attr = "degree_centrality"
            visualizer.plot_graph(
                graph,
                ax=ax,
                node_attr=node_attr,
                node_cmap="plasma",
                node_size=50,
                title=f"Grafo de {name} - Centralidade de Grau",
                show_node_legend=True
            )
        else:
            visualizer.plot_graph(
                graph,
                ax=ax,
                node_size=50,
                title=f"Grafo de {name}",
            )
            
        visualizer.save_figure(fig, os.path.join(viz_dir, f"grafo_{name}.png"))
    
    # Visualiza grafo multicamada
    if len(available_layers) >= 2:
        fig, ax = plt.subplots(figsize=(15, 12))
        
        visualizer.plot_multi_layer_graph(
            multi_graph, 
            ax=ax,
            title="Grafo Multi-Camada",
            node_size=30
        )
        
        visualizer.save_figure(fig, os.path.join(viz_dir, "grafo_multi_camada.png"))
    
    # Visualiza métricas como gráficos de barras
    for name, metric_dict in metrics.items():
        if name == 'multi':
            # Gráfico para métricas do grafo multicamada
            fig, ax = plt.subplots(figsize=(10, 6))
            
            plt.bar(metric_dict.keys(), metric_dict.values())
            plt.title(f"Métricas do Grafo Multi-Camada")
            plt.xlabel("Métrica")
            plt.ylabel("Valor")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(viz_dir, f"metricas_multi.png"))
            plt.close()
        else:
            # Gráficos para métricas básicas
            if 'basic' in metric_dict:
                basic = {k: v for k, v in metric_dict['basic'].items() 
                        if isinstance(v, (int, float)) and v is not None}
                
                if basic:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    plt.bar(basic.keys(), basic.values())
                    plt.title(f"Métricas Básicas - Grafo de {name}")
                    plt.xlabel("Métrica")
                    plt.ylabel("Valor")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(viz_dir, f"metricas_basicas_{name}.png"))
                    plt.close()
    
    # ETAPA 7: EXPORTAÇÃO DOS RESULTADOS
    logger.info("7. Exportação dos resultados")
    
    # Cria diretório para resultados
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Exporta camadas processadas
    for name, layer in processed_layers.items():
        try:
            layer.to_file(os.path.join(results_dir, f"{name}.gpkg"), driver='GPKG')
        except Exception as e:
            logger.warning(f"Não foi possível exportar a camada {name}: {e}")
    
    # Exporta métricas como CSV
    for name, metric_dict in metrics.items():
        if name == 'multi':
            pd.DataFrame([metric_dict]).to_csv(
                os.path.join(results_dir, f"metricas_{name}.csv"), index=False)
        else:
            if 'basic' in metric_dict:
                pd.DataFrame([metric_dict['basic']]).to_csv(
                    os.path.join(results_dir, f"metricas_basicas_{name}.csv"), index=False)
            
            if 'centrality' in metric_dict:
                # Centralidade é um dicionário de dicionários
                centrality_df = pd.DataFrame()
                for metric_name, values in metric_dict['centrality'].items():
                    if values:
                        centrality_df[metric_name] = pd.Series(values)
                
                if not centrality_df.empty:
                    centrality_df.to_csv(
                        os.path.join(results_dir, f"metricas_centralidade_{name}.csv"))
    
    # Tempo total de execução
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Pipeline completo finalizado em {elapsed_time:.2f} segundos")
    
    # Cria um relatório de execução
    with open(os.path.join(output_dir, 'execution_report.txt'), 'w') as f:
        f.write(f"Relatório de Execução do Pipeline Grapho Terrain\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tempo de execução: {elapsed_time:.2f} segundos\n\n")
        
        f.write(f"Camadas processadas: {len(processed_layers)}\n")
        for name in processed_layers:
            f.write(f"  - {name}\n")
        
        f.write(f"\nGrafos criados: {len(graphs)}\n")
        for name, graph in graphs.items():
            f.write(f"  - {name}: {graph.graph.number_of_nodes()} nós, {graph.graph.number_of_edges()} arestas\n")
        
        if len(available_layers) >= 2:
            f.write(f"\nGrafo multicamada:\n")
            f.write(f"  - Camadas: {', '.join(multi_graph.layers.keys())}\n")
            f.write(f"  - Conexões entre camadas: {len(multi_graph.interlayer_edges)}\n")
        
        f.write(f"\nVisualizações geradas: {len(os.listdir(viz_dir))}\n")
        f.write(f"Resultados exportados: {len(os.listdir(results_dir))}\n")
    
    logger.info(f"Relatório de execução salvo em {os.path.join(output_dir, 'execution_report.txt')}")
    
    return {
        'processed_layers': processed_layers,
        'graphs': graphs,
        'multi_graph': multi_graph if len(available_layers) >= 2 else None,
        'metrics': metrics,
        'visualization_dir': viz_dir,
        'results_dir': results_dir
    }


def main():
    """Função principal."""
    # Processa argumentos
    args = parse_arguments()
    
    # Verifica diretórios
    if args.synthetic:
        # Gera dados sintéticos
        data_dir = gerar_dados_sinteticos(args.data)
    else:
        data_dir = args.data
    
    # Executa o pipeline
    executar_pipeline(
        config=args.config,
        data_dir=data_dir,
        output_dir=args.output,
        use_sorocaba=args.sorocaba
    )
    
    logger.info(f"Todos os resultados foram salvos no diretório: {args.output}")
    logger.info("Pipeline completo executado com sucesso!")


if __name__ == "__main__":
    main() 