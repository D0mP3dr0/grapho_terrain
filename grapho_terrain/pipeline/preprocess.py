#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para pré-processamento de camadas geoespaciais.

Este módulo oferece classes para preparar dados geoespaciais
para análise, incluindo filtragem, transformação, normalização
e enriquecimento com atributos derivados.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, nearest_points
from typing import Dict, List, Tuple, Union, Optional, Any

# Importar módulos específicos do projeto
from grapho_terrain.telecommunications.erb import ERB, create_erb_layer
from grapho_terrain.telecommunications.coverage import calculate_coverage_radius


class LayerProcessor:
    """
    Processador de camadas geoespaciais.
    
    Esta classe fornece métodos para transformação, filtragem e enriquecimento
    de camadas geoespaciais, facilitando a integração entre diferentes fontes de dados.
    """
    
    def __init__(self, crs: str = "EPSG:4326"):
        """
        Inicializa o processador de camadas.
        
        Args:
            crs: Sistema de referência de coordenadas padrão (default: "EPSG:4326")
        """
        self.crs = crs
        
    def standardize_layer(self, gdf: gpd.GeoDataFrame, target_crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Padroniza uma camada geoespacial.
        
        Args:
            gdf: GeoDataFrame a ser padronizado
            target_crs: CRS alvo (se None, usa o CRS padrão)
            
        Returns:
            GeoDataFrame padronizado
        """
        if target_crs is None:
            target_crs = self.crs
            
        # Garantir que a geometria é válida
        gdf = gdf.copy()
        gdf['geometry'] = gdf.geometry.buffer(0)
        
        # Converter para o CRS alvo
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
            
        return gdf
    
    def clip_to_boundary(self, gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Recorta uma camada para os limites de uma área.
        
        Args:
            gdf: GeoDataFrame a ser recortado
            boundary: GeoDataFrame com geometria de limite
            
        Returns:
            GeoDataFrame recortado
        """
        if gdf.crs != boundary.crs:
            boundary = boundary.to_crs(gdf.crs)
            
        return gpd.clip(gdf, boundary.geometry.unary_union)
    
    def add_centroid(self, gdf: gpd.GeoDataFrame, inplace: bool = False) -> gpd.GeoDataFrame:
        """
        Adiciona colunas com as coordenadas do centroide.
        
        Args:
            gdf: GeoDataFrame de entrada
            inplace: Se True, modifica o GeoDataFrame original
            
        Returns:
            GeoDataFrame com colunas adicionais
        """
        if not inplace:
            gdf = gdf.copy()
            
        gdf['centroid'] = gdf.geometry.centroid
        gdf['lon'] = gdf.centroid.x
        gdf['lat'] = gdf.centroid.y
        
        return gdf
    
    def calculate_area_density(self, gdf: gpd.GeoDataFrame, 
                              area_column: str = 'area_m2',
                              density_column: str = 'density',
                              inplace: bool = False) -> gpd.GeoDataFrame:
        """
        Calcula área e densidade para geometrias.
        
        Args:
            gdf: GeoDataFrame com geometrias
            area_column: Nome da coluna para armazenar área
            density_column: Nome da coluna para armazenar densidade
            inplace: Se True, modifica o GeoDataFrame original
            
        Returns:
            GeoDataFrame com colunas de área e densidade
        """
        if not inplace:
            gdf = gdf.copy()
            
        # Converter para um CRS projetado para cálculo de área em metros quadrados
        if gdf.crs and gdf.crs.is_geographic:
            temp_gdf = gdf.to_crs(epsg=3857)  # Pseudo-Mercator
            gdf[area_column] = temp_gdf.geometry.area
        else:
            gdf[area_column] = gdf.geometry.area
            
        # Calcular a densidade total
        total_area = gdf[area_column].sum()
        if total_area > 0:
            gdf[density_column] = gdf[area_column] / total_area
            
        return gdf
    
    def add_distance_features(self, target_gdf: gpd.GeoDataFrame, 
                             reference_gdf: gpd.GeoDataFrame,
                             feature_name: str,
                             k_nearest: int = 3,
                             inplace: bool = False) -> gpd.GeoDataFrame:
        """
        Adiciona atributos de distância entre camadas.
        
        Args:
            target_gdf: GeoDataFrame alvo
            reference_gdf: GeoDataFrame de referência
            feature_name: Prefixo para as colunas de distância
            k_nearest: Número de vizinhos mais próximos
            inplace: Se True, modifica o GeoDataFrame original
            
        Returns:
            GeoDataFrame com atributos de distância
        """
        if not inplace:
            target_gdf = target_gdf.copy()
            
        # Garantir que os CRS são iguais
        if target_gdf.crs != reference_gdf.crs:
            reference_gdf = reference_gdf.to_crs(target_gdf.crs)
            
        # Para cada geometria no GDF alvo
        distances = []
        
        for idx, row in target_gdf.iterrows():
            # Calcular distâncias para todas as geometrias de referência
            dists = reference_gdf.geometry.distance(row.geometry)
            
            # Ordenar distâncias e pegar as k menores
            nearest_k = sorted(dists)[:k_nearest]
            
            # Calcular média, min, max das k menores distâncias
            dist_data = {
                f'{feature_name}_nearest_dist': nearest_k[0] if nearest_k else np.nan,
                f'{feature_name}_avg_{k_nearest}_nearest': np.mean(nearest_k) if nearest_k else np.nan,
                f'{feature_name}_max_{k_nearest}_nearest': np.max(nearest_k) if nearest_k else np.nan
            }
            distances.append(dist_data)
            
        # Adicionar colunas ao GDF alvo
        distances_df = pd.DataFrame(distances, index=target_gdf.index)
        return pd.concat([target_gdf, distances_df], axis=1)


class ERBProcessor:
    """
    Processador de dados de Estações Rádio Base (ERBs).
    
    Esta classe fornece métodos específicos para processar, enriquecer e
    preparar dados de ERBs para análise de cobertura e integração com
    outras camadas geoespaciais.
    """
    
    def __init__(self, crs: str = "EPSG:4326"):
        """
        Inicializa o processador de ERBs.
        
        Args:
            crs: Sistema de referência de coordenadas padrão (default: "EPSG:4326")
        """
        self.crs = crs
        self.layer_processor = LayerProcessor(crs)
    
    def load_erb_data(self, filepath: str, 
                     id_col: str = 'id', 
                     lat_col: str = 'latitude', 
                     lon_col: str = 'longitude', 
                     **kwargs) -> gpd.GeoDataFrame:
        """
        Carrega dados de ERBs de um arquivo CSV ou shapefile.
        
        Args:
            filepath: Caminho para o arquivo
            id_col: Nome da coluna de identificador único
            lat_col: Nome da coluna de latitude
            lon_col: Nome da coluna de longitude
            **kwargs: Argumentos adicionais para ERB
            
        Returns:
            GeoDataFrame com ERBs
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.csv':
            # Carregar de CSV
            df = pd.read_csv(filepath)
            return create_erb_layer(df, id_col, lat_col, lon_col, self.crs, **kwargs)
        elif file_ext in ['.shp', '.geojson']:
            # Carregar de arquivo geoespacial
            gdf = gpd.read_file(filepath)
            gdf = self.layer_processor.standardize_layer(gdf)
            
            # Criar objetos ERB
            erbs = []
            for idx, row in gdf.iterrows():
                try:
                    # Extrair coordenadas
                    if id_col in gdf.columns:
                        erb_id = row[id_col]
                    else:
                        erb_id = idx
                        
                    # Criar objeto ERB
                    erb_data = {k: row[k] for k in row.index if k != 'geometry' and not pd.isna(row[k])}
                    erb_data.update(kwargs)
                    
                    erb = ERB(erb_id, row.geometry.y, row.geometry.x, **erb_data)
                    erbs.append(erb)
                except Exception as e:
                    print(f"Erro ao processar ERB {idx}: {e}")
                    
            # Criar GeoDataFrame
            erb_gdf = create_erb_layer(erbs, crs=self.crs)
            return erb_gdf
            
        else:
            raise ValueError(f"Formato de arquivo não suportado: {file_ext}")
    
    def add_coverage_attributes(self, erb_gdf: gpd.GeoDataFrame, 
                              freq_mhz_col: Optional[str] = None,
                              power_dbm_col: Optional[str] = None,
                              gain_dbi_col: Optional[str] = None,
                              default_freq: float = 1800.0,
                              default_power: float = 43.0,
                              default_gain: float = 15.0,
                              loss_exponent: float = 3.5,
                              inplace: bool = False) -> gpd.GeoDataFrame:
        """
        Adiciona atributos de cobertura às ERBs.
        
        Args:
            erb_gdf: GeoDataFrame com ERBs
            freq_mhz_col: Coluna com frequência em MHz
            power_dbm_col: Coluna com potência em dBm
            gain_dbi_col: Coluna com ganho em dBi
            default_freq: Frequência padrão (MHz)
            default_power: Potência padrão (dBm)
            default_gain: Ganho padrão (dBi)
            loss_exponent: Expoente de perda de propagação
            inplace: Se True, modifica o GeoDataFrame original
            
        Returns:
            GeoDataFrame com atributos de cobertura
        """
        if not inplace:
            erb_gdf = erb_gdf.copy()
            
        # Preparar colunas de parâmetros
        if freq_mhz_col and freq_mhz_col not in erb_gdf.columns:
            erb_gdf[freq_mhz_col] = default_freq
            
        if power_dbm_col and power_dbm_col not in erb_gdf.columns:
            erb_gdf[power_dbm_col] = default_power
            
        if gain_dbi_col and gain_dbi_col not in erb_gdf.columns:
            erb_gdf[gain_dbi_col] = default_gain
        
        # Calcular EIRP e raio de cobertura
        erb_gdf['eirp_dbm'] = erb_gdf.apply(
            lambda row: row[power_dbm_col] + row[gain_dbi_col] if 
            (freq_mhz_col and power_dbm_col and gain_dbi_col) else
            default_power + default_gain,
            axis=1
        )
        
        erb_gdf['coverage_radius_m'] = erb_gdf.apply(
            lambda row: calculate_coverage_radius(
                row['eirp_dbm'],
                row[freq_mhz_col] if freq_mhz_col else default_freq,
                loss_exponent=loss_exponent
            ),
            axis=1
        )
        
        return erb_gdf
    
    def create_coverage_geometries(self, erb_gdf: gpd.GeoDataFrame, 
                                  radius_col: str = 'coverage_radius_m',
                                  coverage_col: str = 'coverage_geometry',
                                  inplace: bool = False) -> gpd.GeoDataFrame:
        """
        Cria geometrias de cobertura para ERBs.
        
        Args:
            erb_gdf: GeoDataFrame com ERBs
            radius_col: Coluna com raio de cobertura em metros
            coverage_col: Nome da coluna para geometria de cobertura
            inplace: Se True, modifica o GeoDataFrame original
            
        Returns:
            GeoDataFrame com geometrias de cobertura
        """
        if not inplace:
            erb_gdf = erb_gdf.copy()
            
        # Verificar se o raio está em metros
        if radius_col not in erb_gdf.columns:
            raise ValueError(f"Coluna de raio '{radius_col}' não encontrada")
            
        # Se CRS for geográfico, converter para projetado para buffer em metros
        if erb_gdf.crs and erb_gdf.crs.is_geographic:
            temp_gdf = erb_gdf.to_crs(epsg=3857)  # Pseudo-Mercator
            coverage = temp_gdf.geometry.buffer(temp_gdf[radius_col])
            coverage = gpd.GeoSeries(coverage, crs=temp_gdf.crs).to_crs(erb_gdf.crs)
        else:
            coverage = erb_gdf.geometry.buffer(erb_gdf[radius_col])
            
        erb_gdf[coverage_col] = coverage
        
        return erb_gdf
    
    def enrich_with_terrain(self, erb_gdf: gpd.GeoDataFrame, 
                          terrain_gdf: gpd.GeoDataFrame,
                          elevation_col: str = 'elevation',
                          inplace: bool = False) -> gpd.GeoDataFrame:
        """
        Enriquece ERBs com dados de terreno.
        
        Args:
            erb_gdf: GeoDataFrame com ERBs
            terrain_gdf: GeoDataFrame com dados de terreno/elevação
            elevation_col: Coluna com valor de elevação
            inplace: Se True, modifica o GeoDataFrame original
            
        Returns:
            GeoDataFrame com dados de terreno
        """
        if not inplace:
            erb_gdf = erb_gdf.copy()
            
        # Garantir que os CRSs são iguais
        if erb_gdf.crs != terrain_gdf.crs:
            terrain_gdf = terrain_gdf.to_crs(erb_gdf.crs)
            
        # Realizar join espacial
        joined = gpd.sjoin_nearest(erb_gdf, terrain_gdf[[elevation_col, 'geometry']], how='left')
        
        # Remover colunas de índice do join
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        return joined