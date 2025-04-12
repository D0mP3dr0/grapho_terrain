"""
Construção de grafos geoespaciais a partir de diferentes camadas.

Este módulo fornece classes e funções para criar grafos a partir de
camadas geoespaciais como edifícios, vias, ERBs, etc.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from typing import Dict, List, Optional, Tuple, Union, Any

from ..network.base import GeoGraph
from ..network.feature_graphs import FeatureGeoGraph, MultiLayerFeatureGraph
from ..telecommunications.erb import ERB


class GraphBuilder:
    """Classe para construir grafos a partir de camadas geoespaciais."""
    
    def __init__(self):
        """Inicializa um novo construtor de grafos."""
        pass
    
    def create_road_graph(self, roads: gpd.GeoDataFrame,
                         directed: bool = False,
                         weight_field: Optional[str] = None,
                         simplify: bool = True) -> GeoGraph:
        """
        Cria um grafo de vias a partir de uma camada de vias.
        
        Args:
            roads: GeoDataFrame contendo as vias
            directed: Se True, cria um grafo direcionado
            weight_field: Campo a ser usado como peso das arestas
            simplify: Se True, simplifica o grafo, removendo nós intermediários
            
        Returns:
            GeoGraph representando a rede viária
        """
        # Cria um novo grafo
        G = GeoGraph(directed=directed)
        
        # Adiciona cada via como uma aresta do grafo
        for idx, row in roads.iterrows():
            line = row.geometry
            if not line.is_empty and isinstance(line, LineString):
                # Extrai coordenadas dos nós (início e fim da linha)
                coords = list(line.coords)
                
                if len(coords) >= 2:
                    # Extrai ponto de início e fim
                    start_point = coords[0]
                    end_point = coords[-1]
                    
                    # Atributos para a aresta
                    edge_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                    edge_attrs['geometry'] = line
                    
                    # Define peso da aresta
                    if weight_field and weight_field in roads.columns:
                        edge_attrs['weight'] = row[weight_field]
                    else:
                        edge_attrs['weight'] = line.length
                    
                    # Adiciona nós e aresta
                    G.add_node(start_point, pos=start_point)
                    G.add_node(end_point, pos=end_point)
                    G.add_edge(start_point, end_point, **edge_attrs)
                    
                    # Adiciona nós intermediários se não simplificar
                    if not simplify and len(coords) > 2:
                        for i in range(1, len(coords) - 1):
                            mid_point = coords[i]
                            G.add_node(mid_point, pos=mid_point)
                            
                            # Divide a aresta em segmentos
                            prev_point = coords[i-1]
                            segment = LineString([prev_point, mid_point])
                            
                            # Adiciona aresta para segmento
                            segment_attrs = edge_attrs.copy()
                            segment_attrs['geometry'] = segment
                            segment_attrs['weight'] = segment.length
                            G.add_edge(prev_point, mid_point, **segment_attrs)
        
        return G
    
    def create_building_graph(self, buildings: gpd.GeoDataFrame,
                            connectivity_threshold: float = 50.0,
                            weight_field: Optional[str] = None) -> GeoGraph:
        """
        Cria um grafo de edifícios, conectando edifícios próximos.
        
        Args:
            buildings: GeoDataFrame contendo os edifícios
            connectivity_threshold: Distância máxima para conectar edifícios (metros)
            weight_field: Campo a ser usado como peso das arestas
            
        Returns:
            GeoGraph representando a rede de edifícios
        """
        # Cria um novo grafo
        G = GeoGraph(directed=False)
        
        # Adiciona cada edifício como um nó
        for idx, row in buildings.iterrows():
            if not row.geometry.is_empty:
                # Usa centroide como posição do nó
                centroid = row.geometry.centroid
                pos = (centroid.x, centroid.y)
                
                # Atributos do nó
                node_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                node_attrs['pos'] = pos
                node_attrs['geometry'] = row.geometry
                
                # Adiciona o nó
                G.add_node(idx, **node_attrs)
        
        # Conecta edifícios próximos
        for i, row_i in buildings.iterrows():
            for j, row_j in buildings.iterrows():
                if i < j:  # Evita duplicações e auto-conexões
                    # Calcula distância entre os edifícios
                    dist = row_i.geometry.distance(row_j.geometry)
                    
                    if dist <= connectivity_threshold:
                        # Define peso da aresta
                        if weight_field and weight_field in buildings.columns:
                            weight = (row_i[weight_field] + row_j[weight_field]) / 2
                        else:
                            weight = dist
                        
                        # Cria uma linha entre os centroides
                        line = LineString([
                            row_i.geometry.centroid,
                            row_j.geometry.centroid
                        ])
                        
                        # Adiciona a aresta
                        G.add_edge(i, j, weight=weight, distance=dist, geometry=line)
        
        return G
    
    def create_network_from_points(self, points: gpd.GeoDataFrame,
                                 k_nearest: int = 3,
                                 max_distance: Optional[float] = None,
                                 weight_field: Optional[str] = None) -> GeoGraph:
        """
        Cria um grafo conectando pontos aos k vizinhos mais próximos.
        
        Args:
            points: GeoDataFrame contendo os pontos
            k_nearest: Número de vizinhos mais próximos a conectar
            max_distance: Distância máxima para conectar pontos
            weight_field: Campo a ser usado como peso das arestas
            
        Returns:
            GeoGraph representando a rede de pontos
        """
        # Cria um novo grafo
        G = GeoGraph(directed=False)
        
        # Adiciona cada ponto como um nó
        for idx, row in points.iterrows():
            if not row.geometry.is_empty:
                pos = (row.geometry.x, row.geometry.y)
                
                # Atributos do nó
                node_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                node_attrs['pos'] = pos
                
                # Adiciona o nó
                G.add_node(idx, **node_attrs)
        
        # Encontra e conecta os k vizinhos mais próximos
        for i, row_i in points.iterrows():
            # Calcula distâncias para todos os outros pontos
            distances = []
            for j, row_j in points.iterrows():
                if i != j:
                    dist = row_i.geometry.distance(row_j.geometry)
                    distances.append((j, dist))
            
            # Ordena por distância e pega os k mais próximos
            distances.sort(key=lambda x: x[1])
            nearest_neighbors = distances[:k_nearest]
            
            # Conecta com os vizinhos (respeitando max_distance se definido)
            for j, dist in nearest_neighbors:
                if max_distance is None or dist <= max_distance:
                    # Define peso da aresta
                    if weight_field and weight_field in points.columns:
                        weight = (points.loc[i, weight_field] + points.loc[j, weight_field]) / 2
                    else:
                        weight = dist
                    
                    # Cria uma linha entre os pontos
                    line = LineString([row_i.geometry, points.loc[j].geometry])
                    
                    # Adiciona a aresta se não existir
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=weight, distance=dist, geometry=line)
        
        return G
    
    def create_erb_network(self, erbs: List[ERB],
                         connectivity_type: str = 'distance',
                         max_distance: float = 5000.0,
                         k_nearest: int = 3) -> GeoGraph:
        """
        Cria um grafo de conectividade entre ERBs.
        
        Args:
            erbs: Lista de objetos ERB
            connectivity_type: Tipo de conectividade ('distance', 'knn', 'voronoi')
            max_distance: Distância máxima para conectar ERBs (metros)
            k_nearest: Número de vizinhos mais próximos a conectar
            
        Returns:
            GeoGraph representando a rede de ERBs
        """
        # Cria um GeoDataFrame a partir das ERBs
        erb_points = []
        for i, erb in enumerate(erbs):
            erb_points.append({
                'erb_id': erb.id if hasattr(erb, 'id') else i,
                'operator': erb.operator,
                'technology': erb.technology,
                'height': erb.height,
                'frequency': erb.frequency,
                'geometry': Point(erb.lon, erb.lat)
            })
        
        erb_gdf = gpd.GeoDataFrame(erb_points, crs="EPSG:4326")
        
        # Cria grafo de acordo com o tipo de conectividade
        if connectivity_type == 'knn':
            return self.create_network_from_points(erb_gdf, k_nearest=k_nearest)
        elif connectivity_type == 'voronoi':
            # Implementação de conectividade baseada em diagrama de Voronoi
            # (apenas vizinhos que compartilham bordas de Voronoi)
            # Esta é uma implementação simplificada
            return self.create_network_from_points(erb_gdf, k_nearest=k_nearest)
        else:  # 'distance'
            # Cria um grafo
            G = GeoGraph(directed=False)
            
            # Adiciona nós
            for i, erb in enumerate(erbs):
                node_id = erb.id if hasattr(erb, 'id') else i
                G.add_node(node_id, 
                          pos=(erb.lon, erb.lat),
                          operator=erb.operator,
                          technology=erb.technology,
                          height=erb.height,
                          frequency=erb.frequency)
            
            # Adiciona arestas baseadas em distância
            for i, erb_i in enumerate(erbs):
                node_i = erb_i.id if hasattr(erb_i, 'id') else i
                for j, erb_j in enumerate(erbs):
                    if i < j:  # Evita duplicações
                        node_j = erb_j.id if hasattr(erb_j, 'id') else j
                        
                        # Calcula distância entre ERBs (em metros)
                        # Aproximação simples - em uma implementação real usaria haversine
                        lat_diff = erb_i.lat - erb_j.lat
                        lon_diff = erb_i.lon - erb_j.lon
                        dist_deg = np.sqrt(lat_diff**2 + lon_diff**2)
                        dist_m = dist_deg * 111000  # Conversão aproximada para metros
                        
                        if dist_m <= max_distance:
                            # Cria linha entre as ERBs
                            line = LineString([(erb_i.lon, erb_i.lat), (erb_j.lon, erb_j.lat)])
                            
                            # Adiciona aresta
                            G.add_edge(node_i, node_j, 
                                      weight=dist_m,
                                      distance=dist_m,
                                      line_of_sight=True,  # Simplificação
                                      geometry=line)
            
            return G
    
    def create_multiplex_graph(self, layers: Dict[str, gpd.GeoDataFrame], 
                              layer_connectors: Dict[Tuple[str, str], Dict[str, Any]]) -> MultiLayerFeatureGraph:
        """
        Cria um grafo multicamadas a partir de várias camadas geoespaciais.
        
        Args:
            layers: Dicionário com {nome_camada: geodataframe}
            layer_connectors: Dicionário com regras de conexão entre camadas
                {(camada1, camada2): {'max_dist': 100, 'predicate': 'intersects'}}
            
        Returns:
            MultiLayerFeatureGraph representando a rede multicamadas
        """
        # Cria um grafo multicamadas
        G = MultiLayerFeatureGraph()
        
        # Adiciona cada camada
        for layer_name, gdf in layers.items():
            # Converte para grafo de características
            layer_graph = self._convert_layer_to_feature_graph(layer_name, gdf)
            
            # Adiciona camada ao grafo multicamadas
            G.add_layer(layer_graph, layer_name)
        
        # Adiciona conexões entre camadas
        for (layer1, layer2), connector_params in layer_connectors.items():
            # Verifica se as camadas existem
            if layer1 in G.layers and layer2 in G.layers:
                self._connect_layers(G, layer1, layer2, **connector_params)
        
        return G
    
    def _convert_layer_to_feature_graph(self, layer_name: str, 
                                      gdf: gpd.GeoDataFrame) -> FeatureGeoGraph:
        """
        Converte uma camada geoespacial para um grafo de características.
        
        Args:
            layer_name: Nome da camada
            gdf: GeoDataFrame com a camada
            
        Returns:
            FeatureGeoGraph representando a camada
        """
        # Cria um grafo de características
        G = FeatureGeoGraph(directed=False)
        
        # Para camadas de pontos ou polígonos (usando centroides)
        if all(isinstance(geom, (Point, Polygon)) for geom in gdf.geometry):
            # Adiciona cada característica como um nó
            for idx, row in gdf.iterrows():
                if isinstance(row.geometry, Polygon):
                    pos = (row.geometry.centroid.x, row.geometry.centroid.y)
                else:  # Point
                    pos = (row.geometry.x, row.geometry.y)
                
                # Atributos do nó
                node_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                node_attrs['pos'] = pos
                node_attrs['geometry'] = row.geometry
                node_attrs['layer'] = layer_name
                
                # Adiciona o nó
                G.add_node(idx, **node_attrs)
        
        # Para camadas de linhas
        elif all(isinstance(geom, LineString) for geom in gdf.geometry):
            # Trata como um grafo de vias
            for idx, row in gdf.iterrows():
                line = row.geometry
                if not line.is_empty:
                    # Extrai coordenadas dos nós (início e fim da linha)
                    coords = list(line.coords)
                    
                    if len(coords) >= 2:
                        # Extrai ponto de início e fim
                        start_point = f"{layer_name}_{idx}_start"
                        end_point = f"{layer_name}_{idx}_end"
                        
                        # Atributos para a aresta
                        edge_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                        edge_attrs['geometry'] = line
                        edge_attrs['layer'] = layer_name
                        edge_attrs['weight'] = line.length
                        
                        # Adiciona nós e aresta
                        G.add_node(start_point, pos=coords[0], layer=layer_name)
                        G.add_node(end_point, pos=coords[-1], layer=layer_name)
                        G.add_edge(start_point, end_point, **edge_attrs)
        
        # Cria matrizes de características para o grafo
        G.create_feature_matrices_from_attributes()
        
        return G
    
    def _connect_layers(self, G: MultiLayerFeatureGraph, layer1: str, layer2: str, 
                       max_dist: Optional[float] = None,
                       predicate: Optional[str] = None,
                       k_nearest: int = 1,
                       weight_attr: str = 'distance') -> None:
        """
        Conecta nós entre duas camadas de um grafo multicamadas.
        
        Args:
            G: Grafo multicamadas
            layer1: Nome da primeira camada
            layer2: Nome da segunda camada
            max_dist: Distância máxima para conectar nós
            predicate: Predicado espacial para conexão ('intersects', 'within', etc.)
            k_nearest: Número de vizinhos mais próximos a conectar
            weight_attr: Atributo a ser usado como peso das arestas
        """
        # Extrai os nós das camadas
        layer1_nodes = {n: data for n, data in G.layers[layer1].nodes(data=True)}
        layer2_nodes = {n: data for n, data in G.layers[layer2].nodes(data=True)}
        
        # Conexão baseada em predicado espacial
        if predicate is not None:
            # Cria GeoDataFrames para cada camada
            gdf1 = self._nodes_to_gdf(layer1_nodes, layer1)
            gdf2 = self._nodes_to_gdf(layer2_nodes, layer2)
            
            # Realiza junção espacial
            joined = gpd.sjoin(gdf1, gdf2, how='inner', predicate=predicate)
            
            # Adiciona arestas entre camadas
            for _, row in joined.iterrows():
                node1 = row['node_id']
                node2 = row['index_right']
                
                # Calcula distância entre os nós
                geom1 = layer1_nodes[node1]['geometry']
                geom2 = layer2_nodes[node2]['geometry']
                distance = geom1.distance(geom2)
                
                # Adiciona aresta se dentro da distância máxima
                if max_dist is None or distance <= max_dist:
                    G.add_interlayer_edge(
                        node1, layer1, 
                        node2, layer2, 
                        weight=distance,
                        distance=distance
                    )
        
        # Conexão baseada em k vizinhos mais próximos
        elif k_nearest > 0:
            for node1, data1 in layer1_nodes.items():
                geom1 = data1['geometry']
                
                # Calcula distâncias para todos os nós da camada 2
                distances = []
                for node2, data2 in layer2_nodes.items():
                    geom2 = data2['geometry']
                    distance = geom1.distance(geom2)
                    distances.append((node2, distance))
                
                # Ordena por distância e pega os k mais próximos
                distances.sort(key=lambda x: x[1])
                nearest_neighbors = distances[:k_nearest]
                
                # Adiciona arestas para os vizinhos mais próximos
                for node2, distance in nearest_neighbors:
                    if max_dist is None or distance <= max_dist:
                        G.add_interlayer_edge(
                            node1, layer1, 
                            node2, layer2, 
                            weight=distance,
                            distance=distance
                        )
    
    def _nodes_to_gdf(self, nodes: Dict[Any, Dict[str, Any]], 
                    layer_name: str) -> gpd.GeoDataFrame:
        """
        Converte um dicionário de nós para um GeoDataFrame.
        
        Args:
            nodes: Dicionário de nós {node_id: {attr: value, ...}}
            layer_name: Nome da camada
            
        Returns:
            GeoDataFrame com os nós
        """
        data = []
        for node_id, attrs in nodes.items():
            row = {'node_id': node_id, 'layer': layer_name}
            for key, value in attrs.items():
                if key != 'pos':  # Exclui a posição, pois já temos a geometria
                    row[key] = value
            data.append(row)
        
        return gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")
    
    def create_feature_graph(self, gdf: gpd.GeoDataFrame, 
                           node_features: List[str] = None,
                           edge_features: List[str] = None,
                           connectivity_type: str = 'knn',
                           k_nearest: int = 3,
                           max_distance: Optional[float] = None) -> FeatureGeoGraph:
        """
        Cria um grafo de características a partir de uma camada geoespacial.
        
        Args:
            gdf: GeoDataFrame com a camada
            node_features: Lista de atributos para usar como features dos nós
            edge_features: Lista de atributos para usar como features das arestas
            connectivity_type: Tipo de conectividade ('knn', 'distance')
            k_nearest: Número de vizinhos mais próximos a conectar
            max_distance: Distância máxima para conectar nós
            
        Returns:
            FeatureGeoGraph com matrizes de características
        """
        # Cria um grafo vazio
        G = FeatureGeoGraph(directed=False)
        
        # Para camadas de pontos ou polígonos (usando centroides)
        if all(isinstance(geom, (Point, Polygon)) for geom in gdf.geometry):
            # Adiciona cada característica como um nó
            for idx, row in gdf.iterrows():
                if isinstance(row.geometry, Polygon):
                    pos = (row.geometry.centroid.x, row.geometry.centroid.y)
                else:  # Point
                    pos = (row.geometry.x, row.geometry.y)
                
                # Atributos do nó
                node_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                node_attrs['pos'] = pos
                node_attrs['geometry'] = row.geometry
                
                # Adiciona o nó
                G.add_node(idx, **node_attrs)
            
            # Conecta nós de acordo com o tipo de conectividade
            if connectivity_type == 'knn':
                # Conecta cada nó aos k vizinhos mais próximos
                for i, row_i in gdf.iterrows():
                    # Calcula distâncias para todos os outros nós
                    distances = []
                    for j, row_j in gdf.iterrows():
                        if i != j:
                            dist = row_i.geometry.distance(row_j.geometry)
                            distances.append((j, dist))
                    
                    # Ordena por distância e pega os k mais próximos
                    distances.sort(key=lambda x: x[1])
                    nearest_neighbors = distances[:k_nearest]
                    
                    # Conecta com os vizinhos
                    for j, dist in nearest_neighbors:
                        if max_distance is None or dist <= max_distance:
                            # Cria uma linha entre os pontos
                            if isinstance(row_i.geometry, Polygon):
                                point_i = row_i.geometry.centroid
                            else:
                                point_i = row_i.geometry
                                
                            if isinstance(gdf.loc[j].geometry, Polygon):
                                point_j = gdf.loc[j].geometry.centroid
                            else:
                                point_j = gdf.loc[j].geometry
                                
                            line = LineString([point_i, point_j])
                            
                            # Adiciona a aresta
                            G.add_edge(i, j, weight=dist, distance=dist, geometry=line)
            
            elif max_distance is not None:
                # Conecta nós dentro de uma distância máxima
                for i, row_i in gdf.iterrows():
                    for j, row_j in gdf.iterrows():
                        if i < j:  # Evita duplicações
                            dist = row_i.geometry.distance(row_j.geometry)
                            if dist <= max_distance:
                                # Cria uma linha entre os pontos
                                if isinstance(row_i.geometry, Polygon):
                                    point_i = row_i.geometry.centroid
                                else:
                                    point_i = row_i.geometry
                                    
                                if isinstance(row_j.geometry, Polygon):
                                    point_j = row_j.geometry.centroid
                                else:
                                    point_j = row_j.geometry
                                    
                                line = LineString([point_i, point_j])
                                
                                # Adiciona a aresta
                                G.add_edge(i, j, weight=dist, distance=dist, geometry=line)
        
        # Para camadas de linhas
        elif all(isinstance(geom, LineString) for geom in gdf.geometry):
            # Trata como um grafo de vias
            for idx, row in gdf.iterrows():
                line = row.geometry
                if not line.is_empty:
                    # Extrai coordenadas dos nós (início e fim da linha)
                    coords = list(line.coords)
                    
                    if len(coords) >= 2:
                        # Extrai ponto de início e fim
                        start_point = f"{idx}_start"
                        end_point = f"{idx}_end"
                        
                        # Atributos para a aresta
                        edge_attrs = {k: v for k, v in row.items() if k != 'geometry'}
                        edge_attrs['geometry'] = line
                        edge_attrs['weight'] = line.length
                        
                        # Adiciona nós e aresta
                        G.add_node(start_point, pos=coords[0])
                        G.add_node(end_point, pos=coords[-1])
                        G.add_edge(start_point, end_point, **edge_attrs)
        
        # Configura features para nós
        if node_features:
            for feature in node_features:
                if feature in gdf.columns:
                    # Cria feature para todos os nós
                    feature_values = {}
                    for idx, row in gdf.iterrows():
                        if idx in G.nodes:
                            feature_values[idx] = row[feature]
                        elif f"{idx}_start" in G.nodes:  # Para grafos de linha
                            feature_values[f"{idx}_start"] = row[feature]
                            feature_values[f"{idx}_end"] = row[feature]
                    
                    # Define a feature no grafo
                    G.set_node_features(feature_values, feature)
        
        # Configura features para arestas
        if edge_features:
            for feature in edge_features:
                if feature in gdf.columns:
                    # Para grafos onde as arestas correspondem diretamente às geometrias
                    if all(isinstance(geom, LineString) for geom in gdf.geometry):
                        feature_values = {}
                        for idx, row in gdf.iterrows():
                            start_point = f"{idx}_start"
                            end_point = f"{idx}_end"
                            if G.has_edge(start_point, end_point):
                                feature_values[(start_point, end_point)] = row[feature]
                        
                        # Define a feature no grafo
                        G.set_edge_features(feature_values, feature)
        
        # Cria matrizes de características
        G.create_feature_matrices_from_attributes()
        
        return G 