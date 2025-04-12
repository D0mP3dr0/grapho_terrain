"""
Módulo de visualização para o pipeline de análise geoespacial.

Este módulo fornece classes e funções para visualizar camadas geoespaciais,
gráficos de rede, resultados de análise e outros componentes do pipeline.
"""

import os
import io
import base64
import contextlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import networkx as nx

from shapely.geometry import Point, LineString, Polygon, MultiPolygon


class Visualizer:
    """Classe para visualização de dados geoespaciais e gráficos."""
    
    def __init__(self, output_dir: str = "output", style: str = "whitegrid", 
                 figsize: Tuple[int, int] = (12, 10), dpi: int = 100):
        """
        Inicializa o visualizador.
        
        Args:
            output_dir: Diretório para salvar as visualizações
            style: Estilo do seaborn para os gráficos
            figsize: Tamanho padrão das figuras
            dpi: Resolução das figuras
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Cria diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configura estilo
        sns.set_style(style)
        
        # Cores para diferentes camadas
        self.layer_colors = {
            'buildings': '#1f77b4',
            'roads': '#ff7f0e',
            'water': '#2ca02c',
            'poi': '#d62728',
            'erb': '#9467bd',
            'terrain': '#8c564b',
            'coverage': 'lightblue',
            'sectors': 'purple',
            'graph_nodes': '#e377c2',
            'graph_edges': '#7f7f7f',
            'heatmap': 'viridis'
        }
    
    def create_figure(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[Figure, Axes]:
        """
        Cria uma nova figura e eixos.
        
        Args:
            figsize: Tamanho da figura (largura, altura) em polegadas
            
        Returns:
            Tupla (figura, eixos)
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        return fig, ax
    
    def save_figure(self, fig: Figure, filename: str, close: bool = True) -> str:
        """
        Salva a figura em um arquivo.
        
        Args:
            fig: Figura a ser salva
            filename: Nome do arquivo
            close: Se True, fecha a figura após salvar
            
        Returns:
            Caminho completo para o arquivo salvo
        """
        # Verifica se o nome do arquivo tem extensão
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
            filename += '.png'
            
        # Caminho completo
        filepath = os.path.join(self.output_dir, filename)
        
        # Salva a figura
        fig.savefig(filepath, bbox_inches='tight')
        
        # Fecha a figura se solicitado
        if close:
            plt.close(fig)
            
        return filepath
    
    def plot_layer(self, layer: gpd.GeoDataFrame, 
                  title: str = None,
                  column: str = None,
                  cmap: str = 'viridis',
                  categorical: bool = False,
                  legend: bool = True,
                  alpha: float = 0.7,
                  edgecolor: str = 'black',
                  linewidth: float = 0.5,
                  markersize: int = 5,
                  figsize: Optional[Tuple[int, int]] = None,
                  ax: Optional[Axes] = None,
                  basemap: Optional[gpd.GeoDataFrame] = None,
                  basemap_alpha: float = 0.3,
                  add_labels: bool = False,
                  label_column: str = None,
                  **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza uma camada geoespacial.
        
        Args:
            layer: GeoDataFrame com a camada a visualizar
            title: Título do gráfico
            column: Coluna para colorir os elementos
            cmap: Mapa de cores
            categorical: Se True, trata a coluna como categórica
            legend: Se True, adiciona legenda
            alpha: Transparência dos elementos
            edgecolor: Cor das bordas
            linewidth: Largura das linhas
            markersize: Tamanho dos marcadores (pontos)
            figsize: Tamanho da figura
            ax: Eixos preexistentes para plotar
            basemap: GeoDataFrame opcional para usar como mapa base
            basemap_alpha: Transparência do mapa base
            add_labels: Se True, adiciona rótulos aos elementos
            label_column: Coluna a usar para os rótulos
            **kwargs: Parâmetros adicionais para plot
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria figura se não fornecida
        if ax is None:
            fig, ax = self.create_figure(figsize)
        else:
            fig = ax.figure
        
        # Plota mapa base se fornecido
        if basemap is not None:
            basemap.plot(ax=ax, color='lightgrey', alpha=basemap_alpha)
        
        # Determina o tipo de elementos na camada
        geom_type = layer.geometry.iloc[0].geom_type
        
        # Define parâmetros específicos para cada tipo de geometria
        if geom_type in ['Point', 'MultiPoint']:
            if 'color' not in kwargs and column is None:
                kwargs['color'] = self.layer_colors.get('poi', 'red')
            layer_plot = layer.plot(ax=ax, column=column, cmap=cmap, categorical=categorical,
                                   legend=legend, alpha=alpha, markersize=markersize, **kwargs)
        elif geom_type in ['LineString', 'MultiLineString']:
            if 'color' not in kwargs and column is None:
                kwargs['color'] = self.layer_colors.get('roads', 'orange')
            layer_plot = layer.plot(ax=ax, column=column, cmap=cmap, categorical=categorical,
                                   legend=legend, alpha=alpha, linewidth=linewidth, **kwargs)
        else:  # Polygon, MultiPolygon
            if 'color' not in kwargs and column is None:
                kwargs['color'] = self.layer_colors.get('buildings', 'blue')
            layer_plot = layer.plot(ax=ax, column=column, cmap=cmap, categorical=categorical,
                                   legend=legend, alpha=alpha, edgecolor=edgecolor, 
                                   linewidth=linewidth, **kwargs)
        
        # Adiciona rótulos se solicitado
        if add_labels and label_column:
            if geom_type in ['Point', 'MultiPoint']:
                for idx, row in layer.iterrows():
                    ax.annotate(str(row[label_column]), 
                              xy=(row.geometry.x, row.geometry.y),
                              xytext=(3, 3), textcoords="offset points")
            else:
                # Para geometrias que não são pontos, usa o centróide
                for idx, row in layer.iterrows():
                    centroid = row.geometry.centroid
                    ax.annotate(str(row[label_column]), 
                              xy=(centroid.x, centroid.y),
                              xytext=(3, 3), textcoords="offset points")
        
        # Configura o título
        if title:
            ax.set_title(title)
            
        # Remove os eixos para mapas
        ax.set_axis_off()
        
        return fig, ax
    
    def plot_multiple_layers(self, layers: Dict[str, gpd.GeoDataFrame],
                            title: str = None,
                            figsize: Optional[Tuple[int, int]] = None,
                            colors: Dict[str, str] = None,
                            alphas: Dict[str, float] = None,
                            add_legend: bool = True,
                            **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza múltiplas camadas em um único mapa.
        
        Args:
            layers: Dicionário com nome da camada e GeoDataFrame
            title: Título do gráfico
            figsize: Tamanho da figura
            colors: Dicionário com cores para cada camada
            alphas: Dicionário com transparências para cada camada
            add_legend: Se True, adiciona legenda
            **kwargs: Parâmetros adicionais para plot
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria figura
        fig, ax = self.create_figure(figsize)
        
        # Parâmetros padrão
        if colors is None:
            colors = {}
        if alphas is None:
            alphas = {}
            
        legend_elements = []
        
        # Plota cada camada
        for layer_name, layer_gdf in layers.items():
            if layer_gdf is None or layer_gdf.empty:
                continue
                
            # Determina cor e transparência
            color = colors.get(layer_name, self.layer_colors.get(layer_name, 'blue'))
            alpha = alphas.get(layer_name, 0.7)
            
            # Determina tipo de geometria e plota adequadamente
            geom_type = layer_gdf.geometry.iloc[0].geom_type if not layer_gdf.empty else None
            
            if geom_type in ['Point', 'MultiPoint']:
                layer_gdf.plot(ax=ax, color=color, alpha=alpha, **kwargs)
                legend_elements.append(mpatches.Patch(color=color, alpha=alpha, label=layer_name))
            elif geom_type in ['LineString', 'MultiLineString']:
                layer_gdf.plot(ax=ax, color=color, alpha=alpha, **kwargs)
                legend_elements.append(mpatches.Patch(color=color, alpha=alpha, label=layer_name))
            else:  # Polygon, MultiPolygon
                layer_gdf.plot(ax=ax, color=color, alpha=alpha, edgecolor='black', **kwargs)
                legend_elements.append(mpatches.Patch(color=color, alpha=alpha, label=layer_name))
        
        # Adiciona legenda
        if add_legend and legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
            
        # Configura título
        if title:
            ax.set_title(title)
            
        # Remove eixos
        ax.set_axis_off()
        
        return fig, ax
    
    def plot_graph(self, graph: nx.Graph,
                  title: str = None,
                  figsize: Optional[Tuple[int, int]] = None,
                  node_color: Union[str, List[str]] = 'blue',
                  node_size: Union[int, List[int]] = 50,
                  edge_color: Union[str, List[str]] = 'grey',
                  edge_width: Union[float, List[float]] = 1.0,
                  with_labels: bool = False,
                  font_size: int = 8,
                  pos: Optional[Dict] = None,
                  ax: Optional[Axes] = None,
                  base_layer: Optional[gpd.GeoDataFrame] = None,
                  base_alpha: float = 0.2,
                  **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza um grafo do NetworkX.
        
        Args:
            graph: Grafo do NetworkX
            title: Título do gráfico
            figsize: Tamanho da figura
            node_color: Cor dos nós
            node_size: Tamanho dos nós
            edge_color: Cor das arestas
            edge_width: Largura das arestas
            with_labels: Se True, mostra rótulos dos nós
            font_size: Tamanho da fonte para rótulos
            pos: Posições dos nós (se None, usa posições geográficas)
            ax: Eixos preexistentes para plotar
            base_layer: GeoDataFrame para usar como camada base
            base_alpha: Transparência da camada base
            **kwargs: Parâmetros adicionais para nx.draw
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria figura se não fornecida
        if ax is None:
            fig, ax = self.create_figure(figsize)
        else:
            fig = ax.figure
            
        # Plota camada base se fornecida
        if base_layer is not None:
            base_layer.plot(ax=ax, color='lightgrey', alpha=base_alpha)
            
        # Determina posições dos nós
        if pos is None:
            # Tenta usar posições geográficas dos nós
            pos = {}
            for node, data in graph.nodes(data=True):
                if 'x' in data and 'y' in data:
                    pos[node] = (data['x'], data['y'])
                elif 'longitude' in data and 'latitude' in data:
                    pos[node] = (data['longitude'], data['latitude'])
                elif 'lon' in data and 'lat' in data:
                    pos[node] = (data['lon'], data['lat'])
                elif hasattr(node, 'x') and hasattr(node, 'y'):
                    # Se os nós são objetos com atributos x e y
                    pos[node] = (node.x, node.y)
                else:
                    # Se não tiver coordenadas, usa layout spring
                    pos = nx.spring_layout(graph)
                    break
        
        # Desenha o grafo
        nx.draw(graph, pos=pos, ax=ax,
                node_color=node_color, node_size=node_size,
                edge_color=edge_color, width=edge_width,
                with_labels=with_labels, font_size=font_size,
                **kwargs)
        
        # Configura título
        if title:
            ax.set_title(title)
            
        return fig, ax
    
    def plot_geograph(self, graph, 
                     title: str = None,
                     figsize: Optional[Tuple[int, int]] = None,
                     node_column: str = None,
                     edge_column: str = None,
                     node_cmap: str = 'viridis',
                     edge_cmap: str = 'plasma',
                     node_size_column: str = None,
                     base_node_size: int = 50,
                     add_colorbar: bool = True,
                     ax: Optional[Axes] = None,
                     base_layer: Optional[gpd.GeoDataFrame] = None,
                     base_alpha: float = 0.2,
                     **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza um grafo geográfico com nós e arestas coloridos por atributos.
        
        Args:
            graph: Grafo geográfico (NetworkX ou GeoPandas)
            title: Título do gráfico
            figsize: Tamanho da figura
            node_column: Coluna para colorir os nós
            edge_column: Coluna para colorir as arestas
            node_cmap: Mapa de cores para nós
            edge_cmap: Mapa de cores para arestas
            node_size_column: Coluna para tamanho dos nós
            base_node_size: Tamanho base dos nós
            add_colorbar: Se True, adiciona barra de cores
            ax: Eixos preexistentes para plotar
            base_layer: GeoDataFrame para usar como camada base
            base_alpha: Transparência da camada base
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria figura se não fornecida
        if ax is None:
            fig, ax = self.create_figure(figsize)
        else:
            fig = ax.figure
            
        # Plota camada base se fornecida
        if base_layer is not None:
            base_layer.plot(ax=ax, color='lightgrey', alpha=base_alpha)
            
        # Extrai nós e arestas como GeoDataFrames, se possível
        if hasattr(graph, 'get_node_gdf') and hasattr(graph, 'get_edge_gdf'):
            # Caso seja um objeto com métodos específicos
            nodes_gdf = graph.get_node_gdf()
            edges_gdf = graph.get_edge_gdf()
        elif hasattr(graph, 'nodes_gdf') and hasattr(graph, 'edges_gdf'):
            # Caso tenha atributos com GeoDataFrames
            nodes_gdf = graph.nodes_gdf
            edges_gdf = graph.edges_gdf
        else:
            # Caso contrário, tenta converter de NetworkX
            nodes_gdf, edges_gdf = self._graph_to_gdfs(graph)
            
        # Configura tamanho dos nós
        if node_size_column and node_size_column in nodes_gdf.columns:
            # Normaliza para tamanhos razoáveis
            min_size = base_node_size * 0.5
            max_size = base_node_size * 3.0
            
            # Obtém os valores da coluna
            size_values = nodes_gdf[node_size_column]
            
            # Normaliza entre min_size e max_size
            size_range = max_size - min_size
            if size_values.max() > size_values.min():
                normalized_sizes = min_size + size_range * (size_values - size_values.min()) / (size_values.max() - size_values.min())
            else:
                normalized_sizes = [base_node_size] * len(nodes_gdf)
                
            node_sizes = normalized_sizes
        else:
            node_sizes = base_node_size
        
        # Plota arestas
        if edge_column and edge_column in edges_gdf.columns:
            edges_gdf.plot(ax=ax, column=edge_column, cmap=edge_cmap, 
                         legend=add_colorbar, alpha=0.7, linewidth=1.5,
                         legend_kwds={'shrink': 0.5, 'label': edge_column})
        else:
            edges_gdf.plot(ax=ax, color=self.layer_colors.get('graph_edges', 'grey'), 
                         alpha=0.7, linewidth=1.0)
            
        # Plota nós
        if node_column and node_column in nodes_gdf.columns:
            nodes_gdf.plot(ax=ax, column=node_column, cmap=node_cmap, 
                         markersize=node_sizes, alpha=0.8, 
                         legend=add_colorbar and not (edge_column and edge_column in edges_gdf.columns),
                         legend_kwds={'shrink': 0.5, 'label': node_column})
        else:
            nodes_gdf.plot(ax=ax, color=self.layer_colors.get('graph_nodes', 'blue'), 
                         markersize=node_sizes, alpha=0.8)
            
        # Configura título
        if title:
            ax.set_title(title)
            
        # Remove eixos
        ax.set_axis_off()
        
        return fig, ax
    
    def _graph_to_gdfs(self, graph: nx.Graph) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Converte um grafo NetworkX para GeoDataFrames de nós e arestas.
        
        Args:
            graph: Grafo NetworkX
            
        Returns:
            Tupla (nodes_gdf, edges_gdf)
        """
        # Extrai nós
        nodes_data = []
        for node, data in graph.nodes(data=True):
            node_data = data.copy()
            
            # Tenta obter coordenadas
            x, y = None, None
            if 'x' in data and 'y' in data:
                x, y = data['x'], data['y']
            elif 'longitude' in data and 'latitude' in data:
                x, y = data['longitude'], data['latitude']
            elif 'lon' in data and 'lat' in data:
                x, y = data['lon'], data['lat']
                
            if x is not None and y is not None:
                node_data['geometry'] = Point(x, y)
                node_data['node_id'] = node
                nodes_data.append(node_data)
                
        # Extrai arestas
        edges_data = []
        for u, v, data in graph.edges(data=True):
            edge_data = data.copy()
            
            # Obtém coordenadas dos nós
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            
            # Tenta obter coordenadas do nó u
            ux, uy = None, None
            if 'x' in u_data and 'y' in u_data:
                ux, uy = u_data['x'], u_data['y']
            elif 'longitude' in u_data and 'latitude' in u_data:
                ux, uy = u_data['longitude'], u_data['latitude']
            elif 'lon' in u_data and 'lat' in u_data:
                ux, uy = u_data['lon'], u_data['lat']
                
            # Tenta obter coordenadas do nó v
            vx, vy = None, None
            if 'x' in v_data and 'y' in v_data:
                vx, vy = v_data['x'], v_data['y']
            elif 'longitude' in v_data and 'latitude' in v_data:
                vx, vy = v_data['longitude'], v_data['latitude']
            elif 'lon' in v_data and 'lat' in v_data:
                vx, vy = v_data['lon'], v_data['lat']
                
            if ux is not None and uy is not None and vx is not None and vy is not None:
                edge_data['geometry'] = LineString([(ux, uy), (vx, vy)])
                edge_data['source'] = u
                edge_data['target'] = v
                edges_data.append(edge_data)
                
        # Cria GeoDataFrames
        nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry')
        edges_gdf = gpd.GeoDataFrame(edges_data, geometry='geometry')
        
        return nodes_gdf, edges_gdf
    
    def plot_heatmap(self, points: gpd.GeoDataFrame,
                    value_column: str = None,
                    base_layer: Optional[gpd.GeoDataFrame] = None,
                    title: str = None,
                    cmap: str = 'plasma',
                    figsize: Optional[Tuple[int, int]] = None,
                    radius: int = 100,
                    weights: Optional[List[float]] = None,
                    ax: Optional[Axes] = None,
                    **kwargs) -> Tuple[Figure, Axes]:
        """
        Cria um mapa de calor a partir de pontos.
        
        Args:
            points: GeoDataFrame com pontos
            value_column: Coluna com valores para ponderação
            base_layer: GeoDataFrame para usar como camada base
            title: Título do gráfico
            cmap: Mapa de cores
            figsize: Tamanho da figura
            radius: Raio para o mapa de calor
            weights: Pesos para os pontos (alternativa a value_column)
            ax: Eixos preexistentes para plotar
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria figura se não fornecida
        if ax is None:
            fig, ax = self.create_figure(figsize)
        else:
            fig = ax.figure
            
        # Plota camada base se fornecida
        if base_layer is not None:
            base_layer.plot(ax=ax, color='lightgrey', alpha=0.3)
            
        # Extrai coordenadas
        x = points.geometry.x
        y = points.geometry.y
        
        # Determina pesos
        if weights is None and value_column is not None:
            if value_column in points.columns:
                weights = points[value_column]
                
        # Determina limites do mapa
        if base_layer is not None:
            bounds = base_layer.total_bounds
        else:
            bounds = points.total_bounds
            
        # Cria malha para o mapa de calor
        xi = np.linspace(bounds[0], bounds[2], 100)
        yi = np.linspace(bounds[1], bounds[3], 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Cria mapa de calor
        from scipy.interpolate import griddata
        
        if weights is not None:
            zi = griddata((x, y), weights, (xi, yi), method='cubic')
        else:
            # Sem pesos, usa densidade de pontos
            zi = griddata((x, y), np.ones(len(x)), (xi, yi), method='cubic')
            
        # Plota o mapa de calor
        heatmap = ax.pcolormesh(xi, yi, zi, cmap=cmap, alpha=0.7, **kwargs)
        
        # Adiciona barra de cores
        plt.colorbar(heatmap, ax=ax, label=value_column if value_column else 'Densidade')
        
        # Configura título
        if title:
            ax.set_title(title)
            
        # Remove eixos
        ax.set_axis_off()
        
        return fig, ax
    
    def plot_metric_comparison(self, metrics: Dict[str, Dict[str, float]],
                              title: str = "Comparação de Métricas",
                              figsize: Optional[Tuple[int, int]] = None,
                              kind: str = 'bar',
                              **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza uma comparação de métricas entre diferentes itens.
        
        Args:
            metrics: Dicionário com itens e suas métricas
            title: Título do gráfico
            figsize: Tamanho da figura
            kind: Tipo de gráfico ('bar', 'barh', etc.)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Converte para DataFrame
        df = pd.DataFrame(metrics).T
        
        # Cria figura
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Plota
        df.plot(kind=kind, ax=ax, **kwargs)
        
        # Configura título
        ax.set_title(title)
        
        # Ajusta layout
        plt.tight_layout()
        
        return fig, ax
    
    def plot_distribution(self, data: Union[List[float], pd.Series], 
                         title: str = "Distribuição",
                         xlabel: str = None,
                         ylabel: str = "Frequência",
                         figsize: Optional[Tuple[int, int]] = None,
                         bins: int = 30,
                         kde: bool = True,
                         **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza a distribuição de uma variável.
        
        Args:
            data: Lista ou Series com os dados
            title: Título do gráfico
            xlabel: Rótulo do eixo x
            ylabel: Rótulo do eixo y
            figsize: Tamanho da figura
            bins: Número de bins para histograma
            kde: Se True, adiciona estimativa de densidade kernel
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria figura
        if figsize is None:
            figsize = (10, 6)
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Plota usando seaborn
        sns.histplot(data, bins=bins, kde=kde, ax=ax, **kwargs)
        
        # Configura título e rótulos
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Ajusta layout
        plt.tight_layout()
        
        return fig, ax
    
    def plot_radar_chart(self, metrics: Dict[str, Dict[str, float]],
                        title: str = "Gráfico Radar de Métricas",
                        figsize: Optional[Tuple[int, int]] = None,
                        scale: List[float] = None,
                        **kwargs) -> Tuple[Figure, Axes]:
        """
        Cria um gráfico radar para comparar métricas.
        
        Args:
            metrics: Dicionário com itens e suas métricas
            title: Título do gráfico
            figsize: Tamanho da figura
            scale: Lista com valores máximos para cada métrica
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Converte para DataFrame
        df = pd.DataFrame(metrics).T
        
        # Prepara dados
        categories = list(df.columns)
        N = len(categories)
        
        # Determina a escala
        if scale is None:
            scale = [df[col].max() for col in categories]
            
        # Cria figura
        if figsize is None:
            figsize = (8, 8)
            
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=self.dpi)
        
        # Coordenadas angulares para cada eixo
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fecha o círculo
        
        # Plota para cada item
        for i, (item, row) in enumerate(df.iterrows()):
            values = row.values.flatten().tolist()
            
            # Normaliza pela escala
            values = [values[i] / scale[i] for i in range(len(values))]
            
            values += values[:1]  # Fecha o círculo
            
            # Plota
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=item, **kwargs)
            ax.fill(angles, values, alpha=0.1)
            
        # Configura categorias
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Adiciona eixos em diferentes níveis
        for i in range(1, 6):
            ax.plot(angles, [i/5] * (N+1), color='grey', alpha=0.3, linewidth=0.5, linestyle='dashed')
            
        # Título e legenda
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig, ax
    
    def create_animation(self, layers_sequence: List[Dict[str, gpd.GeoDataFrame]],
                        titles: List[str],
                        output_file: str = "animation.gif",
                        figsize: Optional[Tuple[int, int]] = None,
                        duration: int = 1000,  # ms por frame
                        **kwargs) -> str:
        """
        Cria uma animação a partir de uma sequência de camadas.
        
        Args:
            layers_sequence: Lista de dicionários de camadas para cada frame
            titles: Lista de títulos para cada frame
            output_file: Caminho para o arquivo de saída
            figsize: Tamanho da figura
            duration: Duração de cada frame em milissegundos
            **kwargs: Parâmetros adicionais para plot_multiple_layers
            
        Returns:
            Caminho para o arquivo de animação
        """
        import imageio
        
        frames = []
        
        # Cria cada frame
        for i, (layers, title) in enumerate(zip(layers_sequence, titles)):
            fig, ax = self.plot_multiple_layers(layers, title=title, figsize=figsize, **kwargs)
            
            # Salva frame em buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Lê a imagem do buffer
            frame = imageio.imread(buf)
            frames.append(frame)
            
            # Fecha a figura
            plt.close(fig)
            
        # Cria o caminho completo
        output_path = os.path.join(self.output_dir, output_file)
        
        # Cria a animação
        imageio.mimsave(output_path, frames, duration=duration/1000)
        
        return output_path
    
    def plot_network_metrics(self, metrics: Dict[str, float],
                           title: str = "Métricas da Rede",
                           figsize: Optional[Tuple[int, int]] = None,
                           kind: str = 'bar',
                           color: str = None,
                           **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza métricas de rede.
        
        Args:
            metrics: Dicionário com métricas e valores
            title: Título do gráfico
            figsize: Tamanho da figura
            kind: Tipo de gráfico ('bar', 'barh', etc.)
            color: Cor das barras
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Cria DataFrame
        df = pd.Series(metrics).reset_index()
        df.columns = ['Métrica', 'Valor']
        
        # Cria figura
        if figsize is None:
            figsize = (10, 6)
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Plota
        if kind == 'bar':
            ax.bar(df['Métrica'], df['Valor'], color=color or self.layer_colors.get('graph_nodes'), **kwargs)
        elif kind == 'barh':
            ax.barh(df['Métrica'], df['Valor'], color=color or self.layer_colors.get('graph_nodes'), **kwargs)
        else:
            df.plot(kind=kind, x='Métrica', y='Valor', ax=ax, color=color or self.layer_colors.get('graph_nodes'), **kwargs)
        
        # Formata os valores nas barras
        if kind == 'bar':
            for i, v in enumerate(df['Valor']):
                ax.text(i, v + 0.01 * max(df['Valor']), f"{v:.3f}", ha='center')
        elif kind == 'barh':
            for i, v in enumerate(df['Valor']):
                ax.text(v + 0.01 * max(df['Valor']), i, f"{v:.3f}", va='center')
        
        # Configura título
        ax.set_title(title)
        
        # Ajusta layout
        plt.tight_layout()
        
        return fig, ax
    
    def plot_degree_distribution(self, graph: nx.Graph,
                               title: str = "Distribuição de Grau",
                               figsize: Optional[Tuple[int, int]] = None,
                               log_scale: bool = False,
                               **kwargs) -> Tuple[Figure, Axes]:
        """
        Visualiza a distribuição de grau de um grafo.
        
        Args:
            graph: Grafo NetworkX
            title: Título do gráfico
            figsize: Tamanho da figura
            log_scale: Se True, usa escala logarítmica
            **kwargs: Parâmetros adicionais
            
        Returns:
            Tupla (figura, eixos)
        """
        # Calcula distribuição de grau
        degrees = [d for n, d in graph.degree()]
        unique_degrees = sorted(set(degrees))
        hist = [degrees.count(d) for d in unique_degrees]
        
        # Cria figura
        if figsize is None:
            figsize = (10, 6)
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Plota
        ax.bar(unique_degrees, hist, color=self.layer_colors.get('graph_nodes'), **kwargs)
        
        # Configura título e rótulos
        ax.set_title(title)
        ax.set_xlabel('Grau')
        ax.set_ylabel('Frequência')
        
        # Escala logarítmica se solicitado
        if log_scale:
            ax.set_yscale('log')
            ax.set_xscale('log')
            
        # Ajusta layout
        plt.tight_layout()
        
        return fig, ax 