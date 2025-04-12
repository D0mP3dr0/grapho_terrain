#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemplo de uso do pipeline completo para análise de dados geoespaciais.

Este script demonstra o uso completo do pipeline de processamento, análise
e visualização de dados geoespaciais, incluindo:
1. Carregamento e pré-processamento de camadas geoespaciais
2. Processamento de dados de ERBs
3. Criação de grafos de múltiplas camadas
4. Análise de características da rede
5. Visualização dos resultados
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
import networkx as nx

# Importações do projeto
from grapho_terrain.pipeline.preprocess import LayerProcessor, ERBProcessor
from grapho_terrain.telecommunications.erb import ERB, compute_coverage_radius, compute_eirp
from grapho_terrain.network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
from grapho_terrain.pipeline.visualizer import Visualizer


def create_sample_data():
    """
    Cria dados de amostra para o exemplo, incluindo edificações, vias e ERBs.
    
    Returns:
        dict: Dicionário com GeoDataFrames para edificações, vias e ERBs
    """
    # Cria amostra de edificações (polígonos)
    buildings_data = []
    for i in range(20):
        # Cria coordenadas aleatórias para um edifício
        x = np.random.uniform(-46.65, -46.60)
        y = np.random.uniform(-23.57, -23.54)
        
        # Cria um polígono para representar o edifício
        width = np.random.uniform(0.0005, 0.002)
        height = np.random.uniform(0.0005, 0.002)
        polygon = Polygon([
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ])
        
        # Adiciona atributos
        buildings_data.append({
            'id': f'B{i}',
            'height': np.random.uniform(5, 100),  # altura em metros
            'type': np.random.choice(['residential', 'commercial', 'industrial']),
            'area': width * height * 111000 * 111000,  # área aproximada em m²
            'geometry': polygon
        })
    
    # Cria amostra de vias (linhas)
    roads_data = []
    for i in range(15):
        # Cria coordenadas aleatórias para uma via
        x1 = np.random.uniform(-46.65, -46.60)
        y1 = np.random.uniform(-23.57, -23.54)
        length = np.random.uniform(0.005, 0.02)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Cria uma linha para representar a via
        x2 = x1 + length * np.cos(angle)
        y2 = y1 + length * np.sin(angle)
        line = LineString([(x1, y1), (x2, y2)])
        
        # Adiciona atributos
        roads_data.append({
            'id': f'R{i}',
            'type': np.random.choice(['primary', 'secondary', 'tertiary']),
            'width': np.random.uniform(5, 20),  # largura em metros
            'name': f'Road {i}',
            'geometry': line
        })
    
    # Cria amostra de ERBs (pontos)
    erbs_data = []
    operators = ['Claro', 'Vivo', 'TIM', 'Oi']
    technologies = ['4G', '5G']
    frequency_bands = [700, 850, 1800, 2100, 2600, 3500]
    
    for i in range(10):
        # Cria coordenadas aleatórias para uma ERB
        x = np.random.uniform(-46.65, -46.60)
        y = np.random.uniform(-23.57, -23.54)
        point = Point(x, y)
        
        # Seleciona atributos aleatórios
        operator = np.random.choice(operators)
        technology = np.random.choice(technologies)
        frequency = np.random.choice(frequency_bands)
        power = np.random.uniform(10, 60)  # potência em dBm
        height = np.random.uniform(15, 50)  # altura em metros
        
        # Adiciona atributos
        erbs_data.append({
            'id': f'E{i}',
            'name': f'ERB {i}',
            'operator': operator,
            'technology': technology,
            'frequency': frequency,
            'power': power,
            'antenna_height': height,
            'longitude': x,
            'latitude': y,
            'geometry': point
        })
    
    # Cria GeoDataFrames
    buildings_gdf = gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")
    roads_gdf = gpd.GeoDataFrame(roads_data, crs="EPSG:4326")
    erbs_gdf = gpd.GeoDataFrame(erbs_data, crs="EPSG:4326")
    
    return {
        'buildings': buildings_gdf,
        'roads': roads_gdf,
        'erbs': erbs_gdf
    }


def main():
    """
    Função principal que executa o pipeline completo.
    """
    print("Iniciando exemplo do pipeline completo...")
    
    # Cria diretório de saída
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Carrega dados de amostra
    print("Gerando dados de amostra...")
    data = create_sample_data()
    
    # ----- ETAPA 1: PRÉ-PROCESSAMENTO -----
    print("\n1. Pré-processamento de camadas geoespaciais")
    
    # Pré-processamento de camadas
    layer_processor = LayerProcessor()
    
    # Processa camada de edificações
    buildings_processed = layer_processor.process_layer(
        data['buildings'], 
        layer_type='building',
        create_centroids=True
    )
    print(f"  Camada de edificações processada: {len(buildings_processed)} edifícios")
    
    # Processa camada de vias
    roads_processed = layer_processor.process_layer(
        data['roads'], 
        layer_type='road',
        simplify_tolerance=0.0001
    )
    print(f"  Camada de vias processada: {len(roads_processed)} vias")
    
    # Processa dados de ERBs
    erb_processor = ERBProcessor()
    erbs = erb_processor.dataframe_to_erbs(data['erbs'])
    
    # Calcula raio de cobertura para cada ERB
    for erb in erbs:
        # Calcula o EIRP
        if not hasattr(erb, 'eirp') or erb.eirp is None:
            erb.eirp = compute_eirp(erb.power, gain=17.0)  # Assume ganho de 17 dBi
        
        # Calcula o raio de cobertura
        if not hasattr(erb, 'coverage_radius') or erb.coverage_radius is None:
            erb.coverage_radius = compute_coverage_radius(
                erb.eirp, 
                erb.frequency, 
                sensitivity=-110.0,  # Sensibilidade do receptor em dBm
                terrain_factor=1.0   # Fator de terreno (1.0 = plano)
            )
    
    erbs_processed = erb_processor.erbs_to_dataframe(erbs)
    print(f"  Dados de ERBs processados: {len(erbs)} ERBs")
    
    # ----- ETAPA 2: CRIAÇÃO DE GRAFOS -----
    print("\n2. Criação de grafos")
    
    # Cria grafo de edificações
    building_graph = FeatureGeoGraph()
    building_graph.create_from_points(
        buildings_processed,
        id_col='id',
        feature_cols=['height', 'area', 'type'],
        encode_categorical=True,
        point_geometry_col='centroid'
    )
    print(f"  Grafo de edificações criado: {building_graph.graph.number_of_nodes()} nós, " 
          f"{building_graph.graph.number_of_edges()} arestas")
    
    # Cria grafo de vias
    road_graph = FeatureGeoGraph()
    road_graph.create_from_lines(
        roads_processed,
        id_col='id',
        feature_cols=['width', 'type'],
        encode_categorical=True,
        split_lines=True
    )
    print(f"  Grafo de vias criado: {road_graph.graph.number_of_nodes()} nós, " 
          f"{road_graph.graph.number_of_edges()} arestas")
    
    # Cria grafo de ERBs
    erb_graph = FeatureGeoGraph()
    erb_graph.create_from_points(
        erbs_processed,
        id_col='id',
        feature_cols=['operator', 'technology', 'frequency', 'power', 'antenna_height', 'coverage_radius'],
        encode_categorical=True
    )
    print(f"  Grafo de ERBs criado: {erb_graph.graph.number_of_nodes()} nós, " 
          f"{erb_graph.graph.number_of_edges()} arestas")
    
    # Cria grafo multi-camada
    multi_graph = MultiLayerFeatureGraph()
    multi_graph.add_layer('buildings', building_graph)
    multi_graph.add_layer('roads', road_graph)
    multi_graph.add_layer('erbs', erb_graph)
    
    # Adiciona arestas entre camadas - ERBs para edificações mais próximas
    multi_graph.add_interlayer_edges_knn('erbs', 'buildings', k=3, max_distance=0.01)
    
    # Adiciona arestas entre camadas - Edificações para vias mais próximas
    multi_graph.add_interlayer_edges_knn('buildings', 'roads', k=2, max_distance=0.01)
    
    print(f"  Grafo multi-camada criado com {len(multi_graph.layers)} camadas e " 
          f"{len(multi_graph.interlayer_edges)} arestas entre camadas")
    
    # ----- ETAPA 3: ANÁLISE DE REDE -----
    print("\n3. Análise de rede")
    
    # Análise de métricas para grafo de ERBs
    erb_nx_graph = erb_graph.graph
    
    # Calcular centralidades
    degree_centrality = nx.degree_centrality(erb_nx_graph)
    betweenness_centrality = nx.betweenness_centrality(erb_nx_graph)
    
    # Adicionar centralidades como atributos dos nós
    for node, centrality in degree_centrality.items():
        erb_nx_graph.nodes[node]['degree_centrality'] = centrality
    
    for node, centrality in betweenness_centrality.items():
        erb_nx_graph.nodes[node]['betweenness_centrality'] = centrality
    
    # Estatísticas gerais
    erb_metrics = {
        'num_nodes': erb_nx_graph.number_of_nodes(),
        'num_edges': erb_nx_graph.number_of_edges(),
        'density': nx.density(erb_nx_graph),
        'avg_clustering': nx.average_clustering(erb_nx_graph),
        'avg_degree': np.mean([d for _, d in erb_nx_graph.degree()]),
        'avg_shortest_path': nx.average_shortest_path_length(erb_nx_graph) if nx.is_connected(erb_nx_graph) else None
    }
    
    print("  Métricas do grafo de ERBs:")
    for metric, value in erb_metrics.items():
        print(f"    - {metric}: {value}")
    
    # ----- ETAPA 4: VISUALIZAÇÃO -----
    print("\n4. Visualização dos resultados")
    
    # Cria visualizador
    visualizer = Visualizer(figsize=(12, 10), basemap=True)
    
    # Visualização 1: Mapa de camadas
    print("  Criando visualização de camadas...")
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    # Plota camadas
    visualizer.plot_layer(buildings_processed, ax=ax1, color='lightblue', alpha=0.6, label='Edificações')
    visualizer.plot_layer(roads_processed, ax=ax1, color='gray', alpha=0.8, label='Vias')
    
    # Plota ERBs e cobertura
    visualizer.plot_erb_data(erbs, ax=ax1, plot_coverage=True, by_operator=True, title="Mapa de Camadas e ERBs")
    
    # Salva figura
    visualizer.save_figure(fig1, os.path.join(output_dir, "mapa_camadas.png"))
    
    # Visualização 2: Grafo multi-camada
    print("  Criando visualização do grafo multi-camada...")
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    visualizer.plot_multi_layer_graph(
        multi_graph, 
        ax=ax2,
        title="Grafo Multi-Camada",
        node_size=30
    )
    
    # Salva figura
    visualizer.save_figure(fig2, os.path.join(output_dir, "grafo_multi_camada.png"))
    
    # Visualização 3: Métricas de centralidade do grafo de ERBs
    print("  Criando visualização de métricas de centralidade...")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    visualizer.plot_graph(
        erb_graph,
        ax=ax3,
        node_attr="degree_centrality",
        node_cmap="plasma",
        node_size=100,
        title="Centralidade de Grau das ERBs",
        show_node_legend=True
    )
    
    # Salva figura
    visualizer.save_figure(fig3, os.path.join(output_dir, "centralidade_erbs.png"))
    
    # Visualização 4: Métricas de rede
    print("  Criando visualização de métricas de rede...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Filtra métricas numéricas
    numeric_metrics = {k: v for k, v in erb_metrics.items() if isinstance(v, (int, float)) and v is not None}
    
    visualizer.plot_results(
        numeric_metrics,
        kind='bar',
        ax=ax4,
        title="Métricas do Grafo de ERBs",
        xlabel="Métrica",
        ylabel="Valor"
    )
    
    # Salva figura
    visualizer.save_figure(fig4, os.path.join(output_dir, "metricas_rede.png"))
    
    print("\nTodas as visualizações foram salvas no diretório:", output_dir)
    print("Exemplo de pipeline completo finalizado com sucesso!")


if __name__ == "__main__":
    main() 