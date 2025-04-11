# Notebooks Google Colab para Grapho Terrain

Este diretório contém notebooks para Google Colab que permitem executar análises do projeto Grapho Terrain na nuvem, otimizados para processar operações que consomem muita RAM e VRAM.

## Notebooks Disponíveis

### 1. [erb_analysis.ipynb](erb_analysis.ipynb)

Notebook para análise completa de ERBs (Estações Rádio Base), incluindo:
- Carregamento e geração de dados de exemplo
- Cálculo de cobertura e criação de setores
- Visualização de ERBs e áreas de cobertura
- Análise de redes/grafos das ERBs
- Conversão para PyTorch Geometric
- Criação de grafos multi-camada

### 2. [terrain_analysis.ipynb](terrain_analysis.ipynb)

Notebook para análise de terreno, incluindo:
- Processamento de Modelos Digitais de Elevação (DEM)
- Cálculo de derivadas do terreno (declividade, aspecto, hillshade)
- Análise de visibilidade (viewshed)
- Criação de perfis de terreno
- Integração com análise de ERBs

## Como Utilizar

1. Abra os notebooks no Google Colab clicando nos links acima ou faça upload dos arquivos .ipynb na interface do Colab.

2. Siga as instruções em cada notebook, executando as células em sequência.

3. Para carregar seus próprios dados:
   - Monte seu Google Drive usando a célula fornecida
   - Faça upload de seus dados para o Drive
   - Ajuste os caminhos nos notebooks para apontar para seus arquivos

4. Ajuste os parâmetros conforme necessário:
   - Para conjuntos de dados menores, você pode remover as otimizações de memória
   - Para conjuntos de dados maiores, use as funções de processamento em lotes e amostragem

## Requisitos

Os notebooks instalam automaticamente todas as dependências necessárias, incluindo:
- numpy, pandas, geopandas, matplotlib
- rasterio, shapely, elevation
- PyTorch e PyTorch Geometric

## Otimizações

Os notebooks incluem várias otimizações para lidar com grandes volumes de dados:
1. Processamento em lotes para análise de dados tabulares
2. Processamento de rasters em tiles para operações em DEMs
3. Limpeza de memória após operações intensivas
4. Amostragem de dados para visualizações

## Compartilhamento de Resultados

Os resultados são salvos no diretório `/content/output` e podem ser exportados automaticamente para o Google Drive para compartilhamento posterior. 