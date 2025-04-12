#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline simplificado para Grapho Terrain

Este script implementa um fluxo simplificado mas completo do processo do Grapho Terrain,
garantindo que a execução ocorra do início ao fim, incluindo a visualização do grafo.

Fluxo de execução:
1. Carregamento/geração dos dados de entrada
2. Processamento das camadas geoespaciais
3. Criação de grafos para cada camada
4. Criação de grafo multicamada
5. Visualização dos resultados
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
from shapely.geometry import Point, LineString, Polygon

# Configuração do logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pipeline_simples')


def parse_arguments():
    """Processa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Pipeline simplificado para Grapho Terrain')
    
    parser.add_argument('--data', type=str, default='data',
                       help='Diretório com dados de entrada (padrão: data)')
    
    parser.add_argument('--output', type=str, default='output',
                       help='Diretório para resultados (padrão: output)')
    
    parser.add_argument('--sorocaba', action='store_true',
                       help='Usar exemplo de Sorocaba com ERBs')
    
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


def carregar_dados(data_dir):
    """
    Carrega dados geoespaciais de um diretório.
    
    Args:
        data_dir: Diretório com dados
        
    Returns:
        Dicionário com layers carregadas
    """
    logger.info(f"Carregando dados de {data_dir}")
    
    layers = {}
    
    # Verifica se existem arquivos GPKG
    for file in os.listdir(data_dir):
        if file.endswith('.gpkg'):
            file_path = os.path.join(data_dir, file)
            try:
                layer_name = os.path.splitext(file)[0]
                gdf = gpd.read_file(file_path)
                layers[layer_name] = gdf
                logger.info(f"Camada {layer_name} carregada: {len(gdf)} registros")
            except Exception as e:
                logger.warning(f"Erro ao carregar {file_path}: {e}")
    
    # Se não encontrou nenhum dado, gera dados sintéticos
    if not layers:
        logger.warning("Nenhum dado encontrado. Gerando dados sintéticos...")
        raw_dir = gerar_dados_sinteticos(os.path.dirname(data_dir))
        return carregar_dados(raw_dir)
    
    return layers


def criar_grafo_da_camada(layer, layer_name):
    """
    Cria um grafo a partir de uma camada geoespacial.
    
    Args:
        layer: GeoDataFrame com a camada
        layer_name: Nome da camada
        
    Returns:
        Grafo NetworkX
    """
    logger.info(f"Criando grafo para a camada {layer_name}")
    
    # Cria grafo vazio
    G = nx.Graph(name=layer_name)
    
    # Adiciona nós
    for idx, row in layer.iterrows():
        # Define ID do nó
        if 'id' in row:
            node_id = row['id']
        else:
            node_id = f"{layer_name}_{idx}"
        
        # Obtém coordenadas do centro do objeto
        if row.geometry.geom_type == 'Point':
            x, y = row.geometry.x, row.geometry.y
        else:
            # Para outras geometrias, usa o centroide
            x, y = row.geometry.centroid.x, row.geometry.centroid.y
        
        # Adiciona nó com atributos
        attrs = {k: v for k, v in row.items() if k != 'geometry'}
        attrs['x'] = x
        attrs['y'] = y
        attrs['pos'] = (x, y)  # Para visualização
        attrs['layer'] = layer_name
        
        G.add_node(node_id, **attrs)
    
    # Conecta nós próximos - usa KNN (K vizinhos mais próximos)
    k = min(5, len(G.nodes) - 1)  # no máximo 5 vizinhos ou menos se não tiver nós suficientes
    if k > 0:
        # Cria matriz de coordenadas para cálculo rápido
        coords = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in G.nodes])
        nodes = list(G.nodes())
        
        # Para cada nó, conecta aos K mais próximos
        for i, node_id in enumerate(nodes):
            # Calcula distâncias para todos os outros nós
            distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
            
            # Encontra os K mais próximos (ignora o próprio nó)
            nearest = np.argsort(distances)[1:k+1]
            
            # Adiciona arestas para os K mais próximos
            for j in nearest:
                target_id = nodes[j]
                dist = distances[j]
                G.add_edge(node_id, target_id, weight=dist, distance=dist)
    
    logger.info(f"Grafo criado: {len(G.nodes)} nós, {len(G.edges)} arestas")
    return G


def criar_grafo_multicamada(graphs):
    """
    Cria um grafo multicamada combinando vários grafos.
    
    Args:
        graphs: Dicionário de grafos por camada
        
    Returns:
        Grafo combinado
    """
    logger.info(f"Criando grafo multicamada com {len(graphs)} camadas")
    
    # Cria grafo combinado
    G_multi = nx.Graph(name="grafo_multicamada")
    
    # Adiciona todos os nós e arestas dos grafos individuais
    for layer_name, G in graphs.items():
        for node, attrs in G.nodes(data=True):
            multi_node = f"{layer_name}_{node}"
            G_multi.add_node(multi_node, **attrs)
        
        for u, v, attrs in G.edges(data=True):
            multi_u = f"{layer_name}_{u}"
            multi_v = f"{layer_name}_{v}"
            G_multi.add_edge(multi_u, multi_v, **attrs)
    
    # Conecta camadas se tiver pelo menos duas
    if len(graphs) >= 2:
        # Lista de camadas
        layers = list(graphs.keys())
        
        # Cria conexões entre algumas camadas
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i+1]
            
            logger.info(f"Conectando camadas {layer1} e {layer2}")
            
            # Obtém coordenadas dos nós da camada 1
            nodes1 = [(f"{layer1}_{node}", data['x'], data['y']) 
                     for node, data in graphs[layer1].nodes(data=True)]
            
            # Obtém coordenadas dos nós da camada 2
            nodes2 = [(f"{layer2}_{node}", data['x'], data['y']) 
                     for node, data in graphs[layer2].nodes(data=True)]
            
            # Para cada nó na camada 1, conecta ao nó mais próximo na camada 2
            for node1_id, x1, y1 in nodes1:
                # Calcula distâncias para todos os nós da camada 2
                distances = [(node2_id, np.sqrt((x2-x1)**2 + (y2-y1)**2)) 
                            for node2_id, x2, y2 in nodes2]
                
                # Encontra o nó mais próximo
                closest = min(distances, key=lambda x: x[1])
                
                # Conecta os nós se não estão muito longe
                if closest[1] < 0.01:  # limita a ~1km
                    G_multi.add_edge(node1_id, closest[0], 
                                     weight=closest[1], 
                                     distance=closest[1],
                                     interlayer=True)
    
    logger.info(f"Grafo multicamada criado: {len(G_multi.nodes)} nós, {len(G_multi.edges)} arestas")
    return G_multi


def visualizar_grafo(G, output_path, title="Grafo"):
    """
    Visualiza e salva um grafo.
    
    Args:
        G: Grafo NetworkX
        output_path: Caminho para salvar a imagem
        title: Título do gráfico
    """
    logger.info(f"Visualizando grafo: {title}")
    
    # Configuração da figura
    plt.figure(figsize=(12, 10))
    
    # Extrai posições dos nós
    pos = nx.get_node_attributes(G, 'pos')
    
    # Se não tiver posições, usa layout de força
    if not pos:
        pos = nx.spring_layout(G)
    
    # Cores por camada
    layer_colors = {
        'erbs': 'red',
        'buildings': 'blue',
        'edificacoes': 'blue',
        'roads': 'gray',
        'vias': 'gray'
    }
    
    # Determina cores dos nós
    node_colors = []
    for node in G.nodes():
        layer = G.nodes[node].get('layer', 'default')
        node_colors.append(layer_colors.get(layer, 'green'))
    
    # Determina cores das arestas
    edge_colors = []
    for u, v in G.edges():
        if G.edges[u, v].get('interlayer', False):
            edge_colors.append('orange')
        else:
            edge_colors.append('gray')
    
    # Desenha o grafo
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
        node_size=50,
        node_color=node_colors,
        edge_color=edge_colors,
        width=0.5,
        alpha=0.7
    )
    
    # Adiciona legendas
    plt.title(title)
    plt.axis('off')
    
    # Salva a figura
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Grafo salvo em {output_path}")


def executar_pipeline(data_dir='data', output_dir='output'):
    """
    Executa o pipeline simplificado.
    
    Args:
        data_dir: Diretório com dados de entrada
        output_dir: Diretório para resultados
    """
    start_time = time.time()
    logger.info("Iniciando pipeline simplificado...")
    
    # Cria diretórios de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # ETAPA 1: CARREGAMENTO DE DADOS
    logger.info("1. Carregamento de dados")
    layers = carregar_dados(data_dir)
    
    # ETAPA 2: CRIAÇÃO DE GRAFOS POR CAMADA
    logger.info("2. Criação de grafos por camada")
    graphs = {}
    
    for name, layer in layers.items():
        graphs[name] = criar_grafo_da_camada(layer, name)
    
    # ETAPA 3: CRIAÇÃO DE GRAFO MULTICAMADA
    logger.info("3. Criação de grafo multicamada")
    grafo_multicamada = criar_grafo_multicamada(graphs)
    
    # ETAPA 4: VISUALIZAÇÃO
    logger.info("4. Visualização dos resultados")
    
    # Cria diretório para visualizações
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualiza cada grafo individual
    for name, graph in graphs.items():
        output_path = os.path.join(viz_dir, f"grafo_{name}.png")
        visualizar_grafo(graph, output_path, title=f"Grafo de {name}")
    
    # Visualiza grafo multicamada
    output_path = os.path.join(viz_dir, "grafo_multicamada.png")
    visualizar_grafo(grafo_multicamada, output_path, title="Grafo Multi-Camada")
    
    # ETAPA 5: RELATÓRIO
    logger.info("5. Geração de relatório")
    
    # Tempo total de execução
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cria um relatório de execução
    with open(os.path.join(output_dir, 'execution_report.txt'), 'w') as f:
        f.write(f"Relatório de Execução do Pipeline Simplificado\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tempo de execução: {elapsed_time:.2f} segundos\n\n")
        
        f.write(f"Camadas processadas: {len(layers)}\n")
        for name in layers:
            f.write(f"  - {name}: {len(layers[name])} registros\n")
        
        f.write(f"\nGrafos criados: {len(graphs)}\n")
        for name, graph in graphs.items():
            f.write(f"  - {name}: {graph.number_of_nodes()} nós, {graph.number_of_edges()} arestas\n")
        
        f.write(f"\nGrafo multicamada:\n")
        f.write(f"  - Nós: {grafo_multicamada.number_of_nodes()}\n")
        f.write(f"  - Arestas: {grafo_multicamada.number_of_edges()}\n")
        
        f.write(f"\nVisualizações geradas: {len(os.listdir(viz_dir))}\n")
    
    logger.info(f"Relatório de execução salvo em {os.path.join(output_dir, 'execution_report.txt')}")
    logger.info(f"Pipeline completo finalizado em {elapsed_time:.2f} segundos")
    
    return {
        'layers': layers,
        'graphs': graphs,
        'grafo_multicamada': grafo_multicamada,
        'visualization_dir': viz_dir
    }


def main():
    """Função principal."""
    # Processa argumentos
    args = parse_arguments()
    
    # Verifica diretórios
    if not os.path.exists(args.data):
        logger.warning(f"Diretório de dados {args.data} não encontrado. Gerando dados sintéticos.")
        data_dir = os.path.join(args.data, 'raw')
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        data_dir = gerar_dados_sinteticos(args.data)
    else:
        data_dir = args.data
    
    # Executa o pipeline
    executar_pipeline(data_dir=data_dir, output_dir=args.output)
    
    logger.info(f"Todos os resultados foram salvos no diretório: {args.output}")
    logger.info("Pipeline simplificado executado com sucesso!")


if __name__ == "__main__":
    main() 