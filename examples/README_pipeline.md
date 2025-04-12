# Pipeline Completo do Grapho Terrain

Este guia explica como usar o pipeline do Grapho Terrain para processar dados geoespaciais e gerar grafos de múltiplas camadas de forma contínua, desde o carregamento de dados até a visualização final.

## Versões Disponíveis

O projeto oferece duas versões do pipeline:

1. **Pipeline Simplificado** (`pipeline_simples.py`) - Versão leve que não depende de módulos avançados, ideal para iniciantes ou quando algumas dependências não estão disponíveis.

2. **Pipeline Completo** (`pipeline_completo.py`) - Versão completa com todas as funcionalidades, incluindo análises avançadas de métricas, cobertura de ERBs e exportação de resultados em múltiplos formatos.

## Visão Geral

O pipeline integra todas as funcionalidades do Grapho Terrain em um único fluxo contínuo:

1. **Carregamento de dados** - Leitura de dados geoespaciais como ERBs, edificações e vias
2. **Pré-processamento** - Limpeza e padronização das camadas geoespaciais
3. **Análise de cobertura de ERBs** - Cálculo de cobertura para ERBs
4. **Criação de grafos** - Geração de grafos para cada camada
5. **Grafo multicamada** - Integração das diferentes camadas em um único grafo
6. **Análise de métricas** - Cálculo de métricas de rede
7. **Visualização** - Geração de mapas e visualizações de grafos
8. **Exportação** - Salvamento de resultados processados

## Requisitos

- Python 3.8+
- Pacote Grapho Terrain instalado
- Dependências:
  - geopandas
  - networkx
  - matplotlib
  - numpy
  - pandas
  - shapely

## Modo de Uso

### Pipeline Simplificado (Recomendado para iniciantes)

```bash
# Instale o pacote (se ainda não tiver feito isso)
pip install -e .

# Execute o pipeline simplificado
python examples/pipeline_simples.py

# Especificar diretório de dados
python examples/pipeline_simples.py --data data/meus_dados

# Especificar diretório de saída
python examples/pipeline_simples.py --output output/meu_projeto
```

### Pipeline Completo (Usuários avançados)

```bash
# Execute o pipeline completo com as configurações padrão
python examples/pipeline_completo.py

# Usar configuração personalizada
python examples/pipeline_completo.py --config examples/pipeline_config.json

# Usar exemplo de Sorocaba
python examples/pipeline_completo.py --sorocaba

# Gerar dados sintéticos para testes
python examples/pipeline_completo.py --synthetic
```

## Arquivo de Configuração

O pipeline completo pode ser personalizado através de um arquivo JSON. Um exemplo de configuração é fornecido em `examples/pipeline_config.json`.

```json
{
  "steps": [
    {
      "name": "load_data",
      "enabled": true,
      "params": {
        "data_directory": "data/raw",
        "file_format": "gpkg"
      }
    },
    // Outros passos...
  ]
}
```

## Estrutura de Saída

O pipeline gera a seguinte estrutura de saída:

```
output/
├── visualizations/       # Visualizações geradas
│   ├── camada_erbs.png
│   ├── grafo_erbs.png
│   ├── grafo_multi_camada.png
│   └── ...
├── results/              # Resultados processados (pipeline completo)
│   ├── erbs.gpkg
│   ├── metricas_basicas_erbs.csv
│   └── ...
└── execution_report.txt  # Relatório de execução
```

## Diferenças entre as Versões

### Pipeline Simplificado (`pipeline_simples.py`)

- **Vantagens**:
  - Menor dependência de módulos externos
  - Mais rápido e leve
  - Funciona mesmo sem todos os componentes do Grapho Terrain
  - Gera automaticamente dados de teste se nenhum for encontrado
  
- **Recursos**:
  - Carregamento de dados geoespaciais
  - Criação de grafos para cada camada
  - Criação de grafo multicamada
  - Visualização básica dos grafos
  - Relatório de execução

### Pipeline Completo (`pipeline_completo.py`)

- **Vantagens**:
  - Funcionalidades avançadas
  - Análise detalhada de métricas
  - Processamento de cobertura de ERBs
  - Maior controle sobre parâmetros via configuração
  
- **Recursos adicionais**:
  - Pré-processamento avançado de camadas
  - Análise de cobertura e propagação de sinal
  - Cálculo de métricas complexas de rede
  - Múltiplas visualizações e formatos de exportação
  - Suporte a casos específicos como o de Sorocaba

## Exemplo de Sorocaba

O pipeline completo inclui suporte para o caso específico de ERBs de Sorocaba, com processamento correto das coordenadas:

```bash
python examples/pipeline_completo.py --sorocaba
```

Isso carregará os dados de ERBs de Sorocaba, fará a transformação correta de coordenadas do sistema UTM para WGS84, e executará o pipeline completo.

## Fluxo de Trabalho para Pesquisa Acadêmica

Para usar o pipeline em pesquisas acadêmicas:

1. **Preparação de Dados**:
   - Coloque seus dados na pasta `data/raw`
   - Padronize os nomes de colunas conforme documentação

2. **Execução Inicial**:
   - Comece com o pipeline simplificado para verificar seus dados
   - Execute `python examples/pipeline_simples.py --data data/raw`
   - Verifique as visualizações geradas em `output/visualizations`

3. **Análise Avançada**:
   - Quando seus dados estiverem validados, use o pipeline completo
   - Crie um arquivo de configuração personalizado ou use o padrão
   - Execute `python examples/pipeline_completo.py --config minha_config.json`

4. **Resultados**:
   - Use os arquivos gerados em `output/results` para análises adicionais
   - As visualizações em `output/visualizations` podem ser usadas diretamente em artigos

## Solução de Problemas

### Pipeline Simplificado

Se encontrar problemas com o pipeline simplificado:

1. Verifique se tem as dependências básicas (pandas, geopandas, networkx, matplotlib)
2. Confirme que seus dados estão no formato correto (.gpkg ou .shp)
3. Execute com a flag `--data` apontando diretamente para o diretório dos dados

### Pipeline Completo

Se o pipeline completo parar durante a execução:

1. Verifique o relatório de execução em `output/execution_report.txt`
2. Tente executar apenas etapas específicas do pipeline (usando o arquivo de configuração)
3. Verifique a estrutura e o formato dos seus dados de entrada
4. Como alternativa, use o pipeline simplificado

## Citação

Se usar o Grapho Terrain em sua pesquisa, por favor cite:

```
@software{grapho_terrain,
  author = {UFABC},
  title = {Grapho Terrain: A Python Package for Geospatial Graph Analysis},
  year = {2023},
  url = {https://github.com/ufabc/grapho_terrain}
}
```

## Mais Informações

- Documentação completa: `docs/`
- Exemplos adicionais: `examples/`
- Código fonte: `grapho_terrain/` 