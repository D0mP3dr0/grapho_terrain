#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para visualização de dados geoespaciais e grafos de rede.

Este módulo contém classes e funções para visualizar camadas geoespaciais,
grafos de rede e resultados de análises em mapas interativos e gráficos.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, box

# Tenta importar bibliotecas opcionais
try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class Visualizer:
    """
    Classe para visualização de dados geoespaciais e grafos.
    
    Esta classe fornece métodos para criar visualizações estáticas e
    interativas de camadas geoespaciais, grafos de rede e resultados
    de análises.
    """
    
    def __init__(self, figsize=(12, 8), cmap='viridis', style='whitegrid', output_dir=None):
        """
        Inicializa o visualizador.
        
        Args:
            figsize (tuple): Tamanho padrão das figuras (width, height)
            cmap (str): Nome do colormap padrão do matplotlib
            style (str): Estilo do seaborn para gráficos
            output_dir (str): Diretório para salvar as figuras geradas
        """
        self.figsize = figsize
        self.cmap = cmap
        self.style = style
        self.output_dir = output_dir
        
        # Configura o estilo dos gráficos
        sns.set_style(style)
        
        # Cria o diretório de saída se necessário
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_gdf(self, gdf, column=None, scheme='quantiles', k=5, 
                 cmap=None, alpha=0.8, edgecolor='black', figsize=None, 
                 legend=True, title=None, filename=None, **kwargs):
        """
        Plota um GeoDataFrame usando matplotlib.
        
        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame a ser plotado
            column (str): Nome da coluna para colorir as feições
            scheme (str): Esquema de classificação para valores ('equal_interval',
                         'quantiles', 'natural_breaks')
            k (int): Número de classes para classificação
            cmap (str): Nome do colormap
            alpha (float): Transparência (0-1)
            edgecolor (str): Cor do contorno das feições
            figsize (tuple): Tamanho da figura (width, height)
            legend (bool): Se deve incluir legenda
            title (str): Título do gráfico
            filename (str): Nome do arquivo para salvar a figura
            **kwargs: Argumentos adicionais para gdf.plot()
        
        Returns:
            matplotlib.axes.Axes: Axes do matplotlib com o gráfico
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("O argumento gdf deve ser um GeoDataFrame")
        
        figsize = figsize or self.figsize
        cmap = cmap or self.cmap
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plota o GeoDataFrame
        if column:
            gdf.plot(column=column, scheme=scheme, k=k, cmap=cmap, 
                    alpha=alpha, edgecolor=edgecolor, legend=legend, 
                    ax=ax, **kwargs)
        else:
            gdf.plot(alpha=alpha, edgecolor=edgecolor, ax=ax, **kwargs)
        
        # Configura o título
        if title:
            ax.set_title(title)
        
        # Configura os eixos
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        
        # Adiciona grade
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Salva a figura se especificado
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Figura salva em: {filepath}")
        
        return ax
    
    def plot_multiple_layers(self, layers_dict, figsize=None, title=None, 
                            legend=True, filename=None):
        """
        Plota múltiplas camadas geoespaciais em um único mapa.
        
        Args:
            layers_dict (dict): Dicionário com {nome_camada: gdf, ...}
            figsize (tuple): Tamanho da figura (width, height)
            title (str): Título do gráfico
            legend (bool): Se deve incluir legenda
            filename (str): Nome do arquivo para salvar a figura
        
        Returns:
            matplotlib.axes.Axes: Axes do matplotlib com o gráfico
        """
        figsize = figsize or self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Cria uma lista para a legenda
        legend_elements = []
        
        # Cores para as camadas
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Plota cada camada
        for i, (name, gdf) in enumerate(layers_dict.items()):
            color = colors[i % len(colors)]
            gdf.plot(ax=ax, color=color, alpha=0.7)
            
            # Adiciona à legenda
            if legend:
                if isinstance(gdf.geometry.iloc[0], Point):
                    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                label=name, markerfacecolor=color, markersize=10))
                elif isinstance(gdf.geometry.iloc[0], LineString):
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=name))
                else:
                    legend_elements.append(Patch(facecolor=color, alpha=0.7,
                                                edgecolor='black', label=name))
        
        # Adiciona legenda
        if legend and legend_elements:
            ax.legend(handles=legend_elements, loc='best')
        
        # Configura o título
        if title:
            ax.set_title(title)
        
        # Configura os eixos
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        
        # Adiciona grade
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Salva a figura se especificado
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Figura salva em: {filepath}")
        
        return ax
    
    def plot_graph(self, graph, node_color='blue', edge_color='gray',
                  node_size=50, edge_width=1.0, with_labels=False,
                  node_attr=None, edge_attr=None, figsize=None,
                  title=None, filename=None):
        """
        Plota um grafo espacial.
        
        Args:
            graph: Grafo NetworkX ou FeatureGeoGraph a ser plotado
            node_color (str): Cor dos nós ou nome do atributo para colorir
            edge_color (str): Cor das arestas ou nome do atributo para colorir
            node_size (int): Tamanho dos nós ou nome do atributo para dimensionar
            edge_width (float): Largura das arestas ou nome do atributo
            with_labels (bool): Se deve mostrar rótulos dos nós
            node_attr (str): Atributo dos nós a ser visualizado
            edge_attr (str): Atributo das arestas a ser visualizado
            figsize (tuple): Tamanho da figura (width, height)
            title (str): Título do gráfico
            filename (str): Nome do arquivo para salvar a figura
        
        Returns:
            matplotlib.axes.Axes: Axes do matplotlib com o gráfico
        """
        figsize = figsize or self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Se for FeatureGeoGraph, obtém posições dos nós a partir das geometrias
        if hasattr(graph, 'get_node_positions'):
            pos = graph.get_node_positions()
        else:
            # Para graphs do NetworkX, usa atributo 'pos' dos nós
            pos = nx.get_node_attributes(graph, 'pos')
            if not pos:
                # Se não houver posições, usa um layout spring
                pos = nx.spring_layout(graph)
        
        # Processa cores dos nós
        if isinstance(node_color, str) and node_color in graph.nodes[list(graph.nodes)[0]]:
            node_colors = [graph.nodes[n][node_color] for n in graph.nodes]
            node_color = node_colors
        
        # Processa tamanhos dos nós
        if isinstance(node_size, str) and node_size in graph.nodes[list(graph.nodes)[0]]:
            node_sizes = [graph.nodes[n][node_size] for n in graph.nodes]
            node_size = node_sizes
        
        # Processa cores das arestas
        edge_colors = edge_color
        if isinstance(edge_color, str) and edge_attr:
            edge_colors = [graph.edges[u, v][edge_attr] for u, v in graph.edges]
        
        # Processa larguras das arestas
        edge_widths = edge_width
        if isinstance(edge_width, str) and edge_width in graph.edges[list(graph.edges)[0]]:
            edge_widths = [graph.edges[u, v][edge_width] for u, v in graph.edges]
        
        # Desenha o grafo
        nx.draw_networkx(
            graph,
            pos=pos,
            with_labels=with_labels,
            node_color=node_color,
            edge_color=edge_colors,
            node_size=node_size,
            width=edge_widths,
            alpha=0.7,
            ax=ax
        )
        
        # Configura o título
        if title:
            ax.set_title(title)
        
        # Remove os eixos
        ax.set_axis_off()
        
        # Salva a figura se especificado
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Figura salva em: {filepath}")
        
        return ax
    
    def plot_network_metrics(self, graph, metrics, figsize=None, 
                           title=None, filename=None):
        """
        Visualiza métricas de rede usando histogramas e gráficos de dispersão.
        
        Args:
            graph: Grafo NetworkX ou FeatureGeoGraph
            metrics (dict): Dicionário com métricas calculadas
            figsize (tuple): Tamanho da figura (width, height)
            title (str): Título do gráfico
            filename (str): Nome do arquivo para salvar a figura
        
        Returns:
            List[matplotlib.axes.Axes]: Lista de Axes com os gráficos
        """
        if not metrics:
            raise ValueError("O dicionário de métricas está vazio")
        
        # Determina o número de métricas para criar o layout
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Calcula tamanho da figura
        if figsize is None:
            figsize = (self.figsize[0], self.figsize[1] * n_rows / 2)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        # Plota cada métrica
        for i, (metric_name, metric_values) in enumerate(metrics.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Converte para lista se for um dicionário
                if isinstance(metric_values, dict):
                    metric_values = list(metric_values.values())
                
                # Plota histograma
                sns.histplot(metric_values, ax=ax, kde=True)
                ax.set_title(f'Distribuição de {metric_name}')
                ax.set_xlabel(metric_name)
                ax.set_ylabel('Frequência')
        
        # Remove eixos não utilizados
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Configura o título principal
        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)
        
        # Salva a figura se especificado
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Figura salva em: {filepath}")
        
        return axes
    
    def plot_spatial_metrics(self, gdf, metric_column, figsize=None,
                            title=None, cmap=None, filename=None):
        """
        Visualiza métricas espaciais em um mapa.
        
        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame com dados espaciais
            metric_column (str): Nome da coluna com a métrica a ser visualizada
            figsize (tuple): Tamanho da figura (width, height)
            title (str): Título do gráfico
            cmap (str): Nome do colormap
            filename (str): Nome do arquivo para salvar a figura
        
        Returns:
            matplotlib.axes.Axes: Axes do matplotlib com o gráfico
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("O argumento gdf deve ser um GeoDataFrame")
        
        if metric_column not in gdf.columns:
            raise ValueError(f"A coluna {metric_column} não existe no GeoDataFrame")
        
        figsize = figsize or self.figsize
        cmap = cmap or self.cmap
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plota o GeoDataFrame colorido pela métrica
        gdf.plot(column=metric_column, cmap=cmap, alpha=0.7, 
                edgecolor='black', legend=True, ax=ax)
        
        # Configura o título
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Distribuição espacial de {metric_column}')
        
        # Configura os eixos
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        
        # Adiciona grade
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Salva a figura se especificado
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Figura salva em: {filepath}")
        
        return ax
    
    def create_interactive_map(self, base_location=None, zoom_start=13):
        """
        Cria um mapa interativo usando Folium.
        
        Args:
            base_location (tuple): Coordenadas de centro (latitude, longitude)
            zoom_start (int): Nível de zoom inicial
        
        Returns:
            folium.Map: Mapa interativo
        """
        if not HAS_FOLIUM:
            raise ImportError("Folium não está instalado. Execute 'pip install folium'.")
        
        # Usa localização base ou centro aproximado do Brasil
        if base_location is None:
            base_location = [-15.77972, -47.92972]  # Brasília
        
        # Cria mapa
        m = folium.Map(location=base_location, zoom_start=zoom_start)
        
        # Adiciona controles
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.MeasureControl().add_to(m)
        folium.plugins.MousePosition().add_to(m)
        
        return m
    
    def add_layer_to_map(self, m, gdf, name=None, style_function=None, 
                        highlight_function=None, popup_cols=None, 
                        cluster=False, heatmap=False):
        """
        Adiciona uma camada GeoDataFrame a um mapa Folium.
        
        Args:
            m (folium.Map): Mapa Folium
            gdf (gpd.GeoDataFrame): GeoDataFrame a ser adicionado ao mapa
            name (str): Nome da camada
            style_function (callable): Função para estilizar feições
            highlight_function (callable): Função para estilizar feições em destaque
            popup_cols (list): Lista de colunas para exibir no popup
            cluster (bool): Se deve agrupar pontos em clusters
            heatmap (bool): Se deve criar um mapa de calor (para pontos)
        
        Returns:
            folium.Map: Mapa atualizado
        """
        if not HAS_FOLIUM:
            raise ImportError("Folium não está instalado. Execute 'pip install folium'.")
        
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("O argumento gdf deve ser um GeoDataFrame")
        
        # Converte para WGS84 se necessário
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Prepara nome da camada
        name = name or "Layer"
        
        # Estilo padrão se não for fornecido
        if style_function is None:
            style_function = lambda x: {
                'fillColor': '#3388ff',
                'color': '#333333',
                'weight': 1,
                'fillOpacity': 0.7
            }
        
        # Destaque padrão se não for fornecido
        if highlight_function is None:
            highlight_function = lambda x: {
                'fillColor': '#ff3333',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.9
            }
        
        # Seleciona colunas para popup
        if popup_cols:
            popup_cols = [col for col in popup_cols if col in gdf.columns]
        else:
            popup_cols = [col for col in gdf.columns if col != 'geometry'][:5]  # Limita para as 5 primeiras
        
        # Verifica o tipo de geometria predominante
        geom_types = gdf.geometry.type.value_counts()
        most_common_geom = geom_types.index[0] if len(geom_types) > 0 else None
        
        # Para pontos
        if most_common_geom in ['Point', 'MultiPoint']:
            if cluster:
                # Cria um marker cluster
                marker_cluster = folium.plugins.MarkerCluster(name=name)
                
                for idx, row in gdf.iterrows():
                    popup_html = "<table>"
                    for col in popup_cols:
                        popup_html += f"<tr><th>{col}</th><td>{row[col]}</td></tr>"
                    popup_html += "</table>"
                    
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{name} {idx}"
                    ).add_to(marker_cluster)
                
                marker_cluster.add_to(m)
            
            elif heatmap:
                # Cria um mapa de calor
                heat_data = [[row.geometry.y, row.geometry.x] for _, row in gdf.iterrows()]
                folium.plugins.HeatMap(
                    heat_data,
                    name=name,
                    radius=15,
                    min_opacity=0.5
                ).add_to(m)
                
            else:
                # Adiciona como camada de pontos normal
                gjson = folium.GeoJson(
                    gdf,
                    name=name,
                    style_function=style_function,
                    highlight_function=highlight_function,
                    tooltip=folium.GeoJsonTooltip(fields=popup_cols)
                )
                gjson.add_to(m)
        
        # Para linhas e polígonos
        else:
            # Converte para GeoJSON e adiciona ao mapa
            gjson = folium.GeoJson(
                gdf,
                name=name,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.GeoJsonTooltip(fields=popup_cols)
            )
            gjson.add_to(m)
        
        # Adiciona controle de camadas se não existir
        if not any(isinstance(child, folium.LayerControl) for child in m.get_children()):
            folium.LayerControl().add_to(m)
        
        return m
    
    def save_interactive_map(self, m, filename=None):
        """
        Salva um mapa interativo como arquivo HTML.
        
        Args:
            m (folium.Map): Mapa Folium
            filename (str): Nome do arquivo para salvar (sem extensão)
        
        Returns:
            str: Caminho para o arquivo HTML salvo
        """
        if not HAS_FOLIUM:
            raise ImportError("Folium não está instalado. Execute 'pip install folium'.")
        
        if not filename:
            filename = "interactive_map"
        
        if not filename.endswith(".html"):
            filename += ".html"
        
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
        else:
            filepath = filename
        
        m.save(filepath)
        print(f"Mapa interativo salvo em: {filepath}")
        
        return filepath
    
    def close_all(self):
        """
        Fecha todas as figuras abertas do matplotlib.
        """
        plt.close('all') 