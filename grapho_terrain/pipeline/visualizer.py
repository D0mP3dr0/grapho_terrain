"""
Módulo de visualização para dados geoespaciais e grafos.

Este módulo fornece classes para a visualização de camadas geoespaciais,
grafos, redes de telecomunicação e resultados de análises.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional, Any
import contextily as ctx
from shapely.geometry import Point, LineString, Polygon

from ..core.graph import GeoGraph
from ..network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
from ..telecommunications.erb import ERB


class Visualizer:
    """
    Classe para visualização de dados geoespaciais e grafos.
    
    Esta classe oferece métodos para criar visualizações de camadas geoespaciais,
    grafos, redes de telecomunicação e resultados de análises.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10), 
                 crs: str = "EPSG:4326", basemap: bool = True):
        """
        Inicializa o visualizador.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
            crs: Sistema de coordenadas de referência
            basemap: Se True, adiciona um mapa base às visualizações
        """
        self.figsize = figsize
        self.crs = crs
        self.basemap = basemap
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        
    def create_figure(self, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Cria uma nova figura para visualização.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
        
        Returns:
            Tupla com figura e eixos
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def add_basemap(self, ax: plt.Axes, crs: str = None, source: str = 'CartoDB.Positron', 
                    zoom: int = None) -> plt.Axes:
        """
        Adiciona um mapa base à visualização.
        
        Args:
            ax: Eixo do matplotlib
            crs: Sistema de coordenadas de referência
            source: Fonte do mapa base (provider.style)
            zoom: Nível de zoom (se None, é calculado automaticamente)
        
        Returns:
            Eixo atualizado
        """
        if crs is None:
            crs = self.crs
            
        try:
            # Tenta adicionar o mapa base
            ctx.add_basemap(ax, source=source, crs=crs, zoom=zoom)
        except Exception as e:
            print(f"Erro ao adicionar mapa base: {e}")
            
        return ax
    
    def plot_layer(self, gdf: gpd.GeoDataFrame, ax: plt.Axes = None, 
                  column: str = None, cmap: str = 'viridis', 
                  legend: bool = True, title: str = None, **kwargs) -> plt.Axes:
        """
        Plota uma camada geoespacial.
        
        Args:
            gdf: GeoDataFrame a ser plotado
            ax: Eixo do matplotlib (se None, cria um novo)
            column: Coluna para colorir as geometrias
            cmap: Mapa de cores
            legend: Se True, adiciona uma legenda
            title: Título do gráfico
            **kwargs: Argumentos adicionais para o plot
        
        Returns:
            Eixo com o plot
        """
        if ax is None:
            _, ax = self.create_figure()
            
        # Garante que o GeoDataFrame está no CRS correto
        if gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)
            
        # Plota a camada
        gdf.plot(ax=ax, column=column, cmap=cmap, legend=legend, **kwargs)
        
        # Adiciona título
        if title:
            ax.set_title(title)
            
        # Adiciona mapa base se configurado
        if self.basemap:
            self.add_basemap(ax)
            
        return ax
    
    def plot_points(self, gdf: gpd.GeoDataFrame, ax: plt.Axes = None,
                   column: str = None, color: str = None, cmap: str = 'viridis',
                   marker: str = 'o', size: int = 50,
                   legend: bool = True, title: str = None, **kwargs) -> plt.Axes:
        """
        Plota pontos de um GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame com pontos
            ax: Eixo do matplotlib (se None, cria um novo)
            column: Coluna para colorir os pontos
            color: Cor dos pontos (ignorado se column for especificado)
            cmap: Mapa de cores (usado apenas se column for especificado)
            marker: Símbolo do marcador
            size: Tamanho dos pontos
            legend: Se True, adiciona uma legenda
            title: Título do gráfico
            **kwargs: Argumentos adicionais para o plot
        
        Returns:
            Eixo com o plot
        """
        if ax is None:
            _, ax = self.create_figure()
            
        # Garante que o GeoDataFrame está no CRS correto
        if gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)
            
        # Converte para pontos se necessário
        point_gdf = gdf.copy()
        if not all(geom.geom_type == 'Point' for geom in point_gdf.geometry):
            point_gdf.geometry = point_gdf.geometry.centroid
            
        # Prepara os argumentos de plotagem
        plot_kwargs = kwargs.copy()
        if column is not None:
            # Se uma coluna for especificada, usa ela para colorir
            plot_kwargs.update({
                'column': column,
                'cmap': cmap,
                'legend': legend
            })
            # Remove color se estiver presente
            plot_kwargs.pop('color', None)
        else:
            # Se não houver coluna, usa a cor especificada
            if color is not None:
                plot_kwargs['color'] = color
            plot_kwargs['legend'] = False
            
        # Plota os pontos
        point_gdf.plot(ax=ax, marker=marker, markersize=size, **plot_kwargs)
        
        # Adiciona título
        if title:
            ax.set_title(title)
            
        # Adiciona mapa base se configurado
        if self.basemap:
            self.add_basemap(ax)
            
        return ax
    
    def plot_graph(self, graph: Union[nx.Graph, GeoGraph], ax: plt.Axes = None,
                  node_color: str = 'blue', edge_color: str = 'gray',
                  node_size: int = 30, edge_width: float = 1.0,
                  node_attr: str = None, edge_attr: str = None,
                  node_cmap: str = 'viridis', edge_cmap: str = 'viridis',
                  with_labels: bool = False, title: str = None,
                  show_node_legend: bool = False, show_edge_legend: bool = False,
                  **kwargs) -> plt.Axes:
        """
        Plota um grafo.
        
        Args:
            graph: Grafo NetworkX ou GeoGraph
            ax: Eixo do matplotlib (se None, cria um novo)
            node_color: Cor dos nós
            edge_color: Cor das arestas
            node_size: Tamanho dos nós
            edge_width: Largura das arestas
            node_attr: Atributo dos nós para coloração
            edge_attr: Atributo das arestas para coloração
            node_cmap: Mapa de cores para nós
            edge_cmap: Mapa de cores para arestas
            with_labels: Se True, mostra rótulos dos nós
            title: Título do gráfico
            show_node_legend: Se True, adiciona legenda para nós
            show_edge_legend: Se True, adiciona legenda para arestas
            **kwargs: Argumentos adicionais para o plot
        
        Returns:
            Eixo com o plot
        """
        if ax is None:
            _, ax = self.create_figure()
            
        # Extrai o grafo NetworkX se for um GeoGraph
        if isinstance(graph, (GeoGraph, FeatureGeoGraph)):
            nx_graph = graph.graph
            # Obtém posições dos nós a partir de coordenadas geográficas
            pos = {node: (data.get('x', 0), data.get('y', 0)) 
                  for node, data in nx_graph.nodes(data=True)}
        else:
            nx_graph = graph
            # Tenta obter posições do grafo, ou usa layout spring como fallback
            pos = nx.get_node_attributes(nx_graph, 'pos')
            if not pos:
                pos = nx.spring_layout(nx_graph)
            
        # Prepara cores dos nós
        if node_attr:
            node_values = [nx_graph.nodes[n].get(node_attr, 0) for n in nx_graph.nodes]
            node_vmin = kwargs.get('node_vmin', min(node_values))
            node_vmax = kwargs.get('node_vmax', max(node_values))
            node_sm = plt.cm.ScalarMappable(cmap=node_cmap, 
                                           norm=plt.Normalize(vmin=node_vmin, vmax=node_vmax))
            node_colors = [node_sm.to_rgba(val) for val in node_values]
        else:
            node_colors = node_color
            
        # Prepara cores das arestas
        if edge_attr:
            edge_values = [nx_graph.edges[e].get(edge_attr, 0) for e in nx_graph.edges]
            edge_vmin = kwargs.get('edge_vmin', min(edge_values))
            edge_vmax = kwargs.get('edge_vmax', max(edge_values))
            edge_sm = plt.cm.ScalarMappable(cmap=edge_cmap, 
                                           norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax))
            edge_colors = [edge_sm.to_rgba(val) for val in edge_values]
            edge_widths = [edge_width * (val / max(edge_values)) if val else edge_width for val in edge_values]
        else:
            edge_colors = edge_color
            edge_widths = edge_width
            
        # Plota o grafo
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=node_colors, 
                              node_size=node_size, **kwargs.get('node_kwargs', {}))
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, edge_color=edge_colors, 
                              width=edge_widths, **kwargs.get('edge_kwargs', {}))
        if with_labels:
            nx.draw_networkx_labels(nx_graph, pos, ax=ax, **kwargs.get('label_kwargs', {}))
            
        # Adiciona legenda para nós
        if show_node_legend and node_attr:
            node_sm.set_array([])
            cbar = plt.colorbar(node_sm, ax=ax, orientation='vertical', 
                               label=node_attr, pad=0.1)
            
        # Adiciona legenda para arestas
        if show_edge_legend and edge_attr:
            edge_sm.set_array([])
            cbar = plt.colorbar(edge_sm, ax=ax, orientation='vertical', 
                               label=edge_attr, pad=0.1)
            
        # Adiciona título
        if title:
            ax.set_title(title)
            
        # Remove eixos
        ax.set_axis_off()
            
        return ax
    
    def plot_multi_layer_graph(self, graph: MultiLayerFeatureGraph, ax: plt.Axes = None,
                              title: str = None, layer_colors: Dict[str, str] = None,
                              node_size: int = 30, edge_width: float = 1.0,
                              with_labels: bool = False, show_legend: bool = True,
                              **kwargs) -> plt.Axes:
        """
        Plota um grafo multi-camada.
        
        Args:
            graph: Grafo multi-camada
            ax: Eixo do matplotlib (se None, cria um novo)
            title: Título do gráfico
            layer_colors: Dicionário com cores para cada camada
            node_size: Tamanho dos nós
            edge_width: Largura das arestas
            with_labels: Se True, mostra rótulos dos nós
            show_legend: Se True, adiciona legenda das camadas
            **kwargs: Argumentos adicionais para o plot
        
        Returns:
            Eixo com o plot
        """
        if ax is None:
            _, ax = self.create_figure()
            
        # Define cores para cada camada
        if layer_colors is None:
            layer_colors = {}
            for i, layer_name in enumerate(graph.layers.keys()):
                color_idx = i % len(self.colors)
                layer_colors[layer_name] = self.colors[color_idx]
                
        # Plota cada camada
        for layer_name, layer_graph in graph.layers.items():
            color = layer_colors.get(layer_name, 'blue')
            
            # Extrai o grafo NetworkX
            nx_graph = layer_graph.graph
            
            # Obtém posições dos nós
            pos = {node: (data.get('x', 0), data.get('y', 0)) 
                  for node, data in nx_graph.nodes(data=True)}
            
            # Plota nós e arestas
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_color=color, 
                                  node_size=node_size, label=layer_name,
                                  **kwargs.get('node_kwargs', {}))
            nx.draw_networkx_edges(nx_graph, pos, ax=ax, edge_color=color, 
                                  width=edge_width, alpha=0.7,
                                  **kwargs.get('edge_kwargs', {}))
            
        # Plota arestas entre camadas
        if graph.interlayer_edges:
            for (layer1, node1), (layer2, node2), data in graph.interlayer_edges:
                # Obtém posições dos nós
                pos1 = (graph.layers[layer1].graph.nodes[node1].get('x', 0),
                       graph.layers[layer1].graph.nodes[node1].get('y', 0))
                pos2 = (graph.layers[layer2].graph.nodes[node2].get('x', 0),
                       graph.layers[layer2].graph.nodes[node2].get('y', 0))
                
                # Plota aresta entre camadas
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.3, 
                       linestyle='--', linewidth=0.5)
                
        # Adiciona rótulos se solicitado
        if with_labels:
            for layer_name, layer_graph in graph.layers.items():
                nx_graph = layer_graph.graph
                pos = {node: (data.get('x', 0), data.get('y', 0)) 
                      for node, data in nx_graph.nodes(data=True)}
                nx.draw_networkx_labels(nx_graph, pos, ax=ax, font_size=8,
                                      **kwargs.get('label_kwargs', {}))
                
        # Adiciona legenda
        if show_legend:
            ax.legend()
            
        # Adiciona título
        if title:
            ax.set_title(title)
            
        # Remove eixos
        ax.set_axis_off()
            
        return ax
    
    def plot_erb_data(self, erbs: List[ERB], ax: plt.Axes = None, 
                     plot_coverage: bool = True, by_operator: bool = True,
                     plot_sectors: bool = False, title: str = None,
                     marker: str = '^', size: int = 80, **kwargs) -> plt.Axes:
        """
        Plota dados de ERBs.
        
        Args:
            erbs: Lista de objetos ERB
            ax: Eixo do matplotlib (se None, cria um novo)
            plot_coverage: Se True, plota áreas de cobertura
            by_operator: Se True, diferencia ERBs por operadora
            plot_sectors: Se True, plota setores de cobertura
            title: Título do gráfico
            marker: Símbolo do marcador
            size: Tamanho dos marcadores
            **kwargs: Argumentos adicionais para o plot
        
        Returns:
            Eixo com o plot
        """
        if ax is None:
            _, ax = self.create_figure()
            
        # Converte para GeoDataFrame
        points = []
        coverages = []
        sectors = []
        
        for erb in erbs:
            # Cria ponto para a ERB
            point = Point(erb.longitude, erb.latitude)
            point_data = {
                'id': erb.id,
                'name': erb.name,
                'operator': erb.operator,
                'technology': erb.technology,
                'frequency': erb.frequency,
                'geometry': point
            }
            points.append(point_data)
            
            # Cria polígono de cobertura se solicitado
            if plot_coverage and erb.coverage_radius:
                coverage_geom = point.buffer(erb.coverage_radius / 111000)  # Conversão aproximada de graus para km
                coverage_data = {
                    'erb_id': erb.id,
                    'operator': erb.operator,
                    'radius': erb.coverage_radius,
                    'geometry': coverage_geom
                }
                coverages.append(coverage_data)
                
            # Cria setores de cobertura se solicitado
            if plot_sectors and erb.coverage_radius and erb.azimuth is not None:
                # Assumindo que o azimuth é o ângulo central do setor, com 120° de abertura
                azimuth = erb.azimuth
                angle_width = 120  # Largura do setor em graus
                start_angle = (azimuth - angle_width/2) % 360
                end_angle = (azimuth + angle_width/2) % 360
                
                # Cria geometria do setor
                sector_geom = self._create_sector(
                    erb.longitude, erb.latitude, 
                    erb.coverage_radius / 111000,  # Conversão aproximada de graus para km
                    start_angle, end_angle
                )
                
                sector_data = {
                    'erb_id': erb.id,
                    'operator': erb.operator,
                    'azimuth': azimuth,
                    'geometry': sector_geom
                }
                sectors.append(sector_data)
                
        # Cria GeoDataFrames
        points_gdf = gpd.GeoDataFrame(points, crs=self.crs)
        
        # Plota áreas de cobertura
        if coverages:
            coverage_gdf = gpd.GeoDataFrame(coverages, crs=self.crs)
            if by_operator:
                # Agrupa por operadora
                for operator, group in coverage_gdf.groupby('operator'):
                    color_idx = list(coverage_gdf['operator'].unique()).index(operator) % len(self.colors)
                    color = self.colors[color_idx]
                    group.plot(ax=ax, color=color, alpha=0.2, label=f"{operator} (cobertura)")
            else:
                coverage_gdf.plot(ax=ax, color='blue', alpha=0.2, label="Cobertura")
                
        # Plota setores
        if sectors:
            sectors_gdf = gpd.GeoDataFrame(sectors, crs=self.crs)
            if by_operator:
                # Agrupa por operadora
                for operator, group in sectors_gdf.groupby('operator'):
                    color_idx = list(sectors_gdf['operator'].unique()).index(operator) % len(self.colors)
                    color = self.colors[color_idx]
                    group.plot(ax=ax, color=color, alpha=0.4, label=f"{operator} (setor)")
            else:
                sectors_gdf.plot(ax=ax, color='green', alpha=0.4, label="Setores")
                
        # Plota ERBs
        if by_operator:
            # Agrupa por operadora
            for operator, group in points_gdf.groupby('operator'):
                color_idx = list(points_gdf['operator'].unique()).index(operator) % len(self.colors)
                color = self.colors[color_idx]
                group.plot(ax=ax, color=color, marker=marker, markersize=size, 
                          label=operator, **kwargs)
        else:
            points_gdf.plot(ax=ax, color='red', marker=marker, markersize=size, 
                           label='ERBs', **kwargs)
            
        # Adiciona mapa base se configurado
        if self.basemap:
            self.add_basemap(ax)
            
        # Adiciona legenda
        ax.legend()
            
        # Adiciona título
        if title:
            ax.set_title(title)
            
        return ax
    
    def _create_sector(self, lon: float, lat: float, radius: float, 
                      start_angle: float, end_angle: float, 
                      num_points: int = 20) -> Polygon:
        """
        Cria uma geometria de setor circular.
        
        Args:
            lon: Longitude do centro
            lat: Latitude do centro
            radius: Raio do setor
            start_angle: Ângulo inicial em graus
            end_angle: Ângulo final em graus
            num_points: Número de pontos para aproximar o arco
        
        Returns:
            Polígono representando o setor
        """
        # Converte ângulos para radianos
        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)
        
        # Trata casos especiais (cruzamento do norte)
        if end_angle < start_angle:
            end_rad += 2 * np.pi
            
        # Gera pontos do arco
        angles = np.linspace(start_rad, end_rad, num_points)
        points = [(lon + radius * np.sin(angle), lat + radius * np.cos(angle)) 
                 for angle in angles]
        
        # Adiciona centro e fecha o polígono
        points = [(lon, lat)] + points
        
        return Polygon(points)
    
    def plot_results(self, results: Dict[str, Any], kind: str = 'bar',
                    ax: plt.Axes = None, title: str = None, 
                    xlabel: str = None, ylabel: str = None,
                    **kwargs) -> plt.Axes:
        """
        Plota resultados de análise.
        
        Args:
            results: Dicionário com resultados
            kind: Tipo de gráfico ('bar', 'line', 'scatter', etc.)
            ax: Eixo do matplotlib (se None, cria um novo)
            title: Título do gráfico
            xlabel: Rótulo do eixo x
            ylabel: Rótulo do eixo y
            **kwargs: Argumentos adicionais para o plot
        
        Returns:
            Eixo com o plot
        """
        if ax is None:
            _, ax = self.create_figure()
            
        # Converte para DataFrame se for um dicionário simples
        if isinstance(results, dict) and not isinstance(next(iter(results.values())), (dict, list)):
            df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
            df.index.name = xlabel or 'Métrica'
            df.plot(kind=kind, ax=ax, **kwargs)
        elif isinstance(results, pd.DataFrame):
            # Se já for um DataFrame
            results.plot(kind=kind, ax=ax, **kwargs)
        else:
            # Tenta converter listas ou dicionários aninhados
            try:
                df = pd.DataFrame(results)
                df.plot(kind=kind, ax=ax, **kwargs)
            except:
                raise ValueError("Formato de resultados não suportado")
                
        # Adiciona rótulos
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Adiciona título
        if title:
            ax.set_title(title)
            
        return ax
    
    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300, 
                   bbox_inches: str = 'tight', **kwargs) -> None:
        """
        Salva a figura em um arquivo.
        
        Args:
            fig: Figura do matplotlib
            filepath: Caminho do arquivo
            dpi: Resolução em pontos por polegada
            bbox_inches: Modo de ajuste de bordas
            **kwargs: Argumentos adicionais para salvar
        """
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Salva a figura
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Figura salva em: {filepath}")
    
    def close_figure(self, fig: plt.Figure = None) -> None:
        """
        Fecha a figura.
        
        Args:
            fig: Figura do matplotlib (se None, fecha a figura atual)
        """
        if fig:
            plt.close(fig)
        else:
            plt.close() 