#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemplo de utilização do módulo de ERBs para dados de Sorocaba.

Este script demonstra como carregar e processar dados de ERBs (Estações Rádio Base)
da cidade de Sorocaba, utilizando o sistema de coordenadas adequado para a região.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Importações do projeto
from grapho_terrain.telecommunications.erb import (
    carregar_dados_erbs_personalizado,
    create_erb_layer
)
from grapho_terrain.telecommunications.coverage import calculate_signal_strength
from grapho_terrain.pipeline.visualizer import Visualizer

def main():
    """
    Função principal para demonstrar o carregamento e visualização de dados de ERBs de Sorocaba.
    """
    print("Exemplo de carregamento de ERBs de Sorocaba")
    
    # Criar diretório para saída
    output_dir = "output/sorocaba"
    os.makedirs(output_dir, exist_ok=True)
    
    # Caminho para o arquivo CSV com dados de ERBs de Sorocaba
    # Substitua pelo caminho real do seu arquivo
    arquivo_csv = "data/sorocaba_erbs.csv"
    
    # Verifique se o arquivo existe, caso contrário, crie um exemplo
    if not os.path.exists(arquivo_csv):
        print(f"Arquivo {arquivo_csv} não encontrado. Criando dados de exemplo...")
        criar_dados_exemplo(arquivo_csv)
    
    # Carregue os dados com o sistema de coordenadas correto para Sorocaba
    print("Carregando dados de ERBs com o sistema de coordenadas correto para Sorocaba...")
    erbs = carregar_dados_erbs_personalizado(
        arquivo_csv,
        crs_origem="EPSG:31983",  # SIRGAS 2000 / UTM zone 23S (comum em Sorocaba)
        crs_destino="EPSG:4326",  # WGS84 (padrão global usado pelo sistema)
        lon_col='longitude',
        lat_col='latitude'
    )
    
    # Exiba informações sobre os dados carregados
    print(f"Dados carregados com sucesso: {len(erbs.erbs)} ERBs encontradas")
    
    # Converta para GeoDataFrame para visualização
    gdf_erbs = erbs.to_geodataframe()
    
    # Crie um visualizador
    visualizer = Visualizer(figsize=(12, 10), basemap=True)
    
    # Visualize as ERBs
    print("Criando visualização de ERBs...")
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    # Plota ERBs
    visualizer.plot_points(gdf_erbs, ax=ax1, color='red', size=50, alpha=0.7, label='ERBs')
    ax1.set_title("ERBs de Sorocaba")
    
    # Salva figura
    visualizer.save_figure(fig1, os.path.join(output_dir, "erbs_sorocaba.png"))
    
    # Criar setores de cobertura
    print("Criando setores de cobertura...")
    gdf_setores = erbs.create_coverage_sectors(tipo_area='urbana')
    
    # Visualiza ERBs com setores de cobertura
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    # Plota setores de cobertura
    visualizer.plot_layer(gdf_setores, ax=ax2, column='operadora', alpha=0.5, legend=True)
    
    # Plota ERBs
    visualizer.plot_points(gdf_erbs, ax=ax2, color='black', size=20, alpha=1.0, label='ERBs')
    
    ax2.set_title("Cobertura de ERBs em Sorocaba")
    
    # Salva figura
    visualizer.save_figure(fig2, os.path.join(output_dir, "cobertura_erbs_sorocaba.png"))
    
    print(f"Visualizações salvas no diretório: {output_dir}")
    print("Exemplo concluído com sucesso!")

def criar_dados_exemplo(arquivo_csv):
    """
    Cria dados de exemplo para ERBs em Sorocaba caso o arquivo real não exista.
    
    Parameters
    ----------
    arquivo_csv : str
        Caminho onde o arquivo CSV será salvo
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(arquivo_csv), exist_ok=True)
    
    # Coordenadas aproximadas do centro de Sorocaba (no sistema SIRGAS 2000 / UTM zone 23S)
    # Nota: Estas são coordenadas em SIRGAS 2000 / UTM zone 23S (EPSG:31983)
    centro_x = 236000  # longitude/easting aproximada em UTM
    centro_y = 7397000  # latitude/northing aproximada em UTM
    
    # Cria dados fictícios de ERBs
    operadoras = ['Claro', 'Vivo', 'TIM', 'Oi']
    tecnologias = ['4G', '5G']
    frequencias = [700, 850, 1800, 2100, 2600, 3500]
    
    erbs_data = []
    for i in range(15):
        # Gera coordenadas aleatórias em torno do centro de Sorocaba
        x = centro_x + np.random.uniform(-5000, 5000)
        y = centro_y + np.random.uniform(-5000, 5000)
        
        # Seleciona atributos aleatórios
        operadora = np.random.choice(operadoras)
        tecnologia = np.random.choice(tecnologias)
        frequencia = np.random.choice(frequencias)
        potencia = np.random.uniform(10, 40)  # potência em watts
        ganho = np.random.uniform(10, 18)  # ganho em dBi
        azimute = np.random.uniform(0, 360)  # azimute em graus
        altura = np.random.uniform(15, 45)  # altura em metros
        
        # Adiciona os dados da ERB
        erbs_data.append({
            'id': f'ERB{i+1:03d}',
            'nome': f'ERB-{operadora}-{i+1}',
            'operadora': operadora,
            'tecnologia': tecnologia,
            'freq_mhz': frequencia,
            'potencia_watts': potencia,
            'ganho_antena': ganho,
            'azimute': azimute,
            'altura_m': altura,
            'longitude': x,
            'latitude': y
        })
    
    # Cria DataFrame com os dados
    df = pd.DataFrame(erbs_data)
    
    # Salva como CSV
    df.to_csv(arquivo_csv, index=False)
    print(f"Dados de exemplo criados e salvos em {arquivo_csv}")

if __name__ == "__main__":
    main() 