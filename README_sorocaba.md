# Instruções para Dados de ERBs de Sorocaba

## Problema de Coordenadas

Ao trabalhar com dados geoespaciais de Sorocaba no Grapho Terrain, você pode enfrentar um problema onde as coordenadas são interpretadas incorretamente e as localizações aparecem em São Paulo em vez de Sorocaba. Isso acontece porque:

1. Os dados originais de Sorocaba provavelmente estão no sistema de coordenadas SIRGAS 2000 / UTM zone 23S (EPSG:31983), comum no Brasil
2. O Grapho Terrain usa por padrão o sistema WGS84 (EPSG:4326) para todas as operações

## Solução

Para resolver este problema, foi adicionada uma nova função chamada `carregar_dados_erbs_personalizado()` ao módulo `grapho_terrain/telecommunications/erb.py`, que permite especificar o sistema de coordenadas de origem e destino ao carregar seus dados.

### Como usar a solução

```python
from grapho_terrain.telecommunications.erb import carregar_dados_erbs_personalizado

# Carregar dados de Sorocaba
erbs = carregar_dados_erbs_personalizado(
    'caminho_para_seu_arquivo.csv',
    crs_origem="EPSG:31983",  # SIRGAS 2000 / UTM zone 23S (comum em Sorocaba)
    crs_destino="EPSG:4326"   # WGS84 (padrão global)
)
```

### Sistemas de coordenadas comuns no Brasil

Se seus dados não estiverem no SIRGAS 2000, você pode tentar outros sistemas comuns no Brasil:

- EPSG:31983 - SIRGAS 2000 / UTM zone 23S (recomendado para Sorocaba)
- EPSG:29193 - SAD69 / UTM zone 23S (mais antigo, mas ainda comum)
- EPSG:22523 - Corrego Alegre / UTM zone 23S (sistemas mais antigos)

### Exemplo completo

Um exemplo completo foi criado em `examples/sorocaba/erb_sorocaba.py` que demonstra:

1. Como carregar dados no sistema de coordenadas correto
2. Como criar visualizações de ERBs e cobertura em Sorocaba
3. Como processar dados com as funções de análise de cobertura

Para executar o exemplo:

```bash
# Instale o pacote (se ainda não tiver feito isso)
pip install -e .

# Execute o exemplo
python examples/sorocaba/erb_sorocaba.py
```

## Verificando se a transformação funcionou

Para verificar se as coordenadas estão corretas após a transformação:

1. Observe a localização das ERBs no mapa gerado em `output/sorocaba/erbs_sorocaba.png`
2. Verifique se os pontos estão próximos da região de Sorocaba
3. Se necessário, ajuste o sistema de coordenadas de origem (parâmetro `crs_origem`) até encontrar o que melhor representa seus dados

Se precisar de mais assistência, entre em contato com a equipe de desenvolvimento. 