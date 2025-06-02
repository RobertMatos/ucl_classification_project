# UCL Player Position Classifier ⚽️

Este projeto aplica técnicas de aprendizado de máquina para prever a posição (zagueiro, meia, atacante etc.) de jogadores da UEFA Champions League com base em estatísticas como gols, assistências e minutos jogados.

## 📁 Estrutura do Projeto
```
ucl_classification_project/
│
├── data/
│   ├── raw/
│   │   └── key_stats.csv       # Dados originais dos jogadores
│   └── processed/              # Dados limpos
│
├── results/
│   └── plots/                  # Gráficos gerados (matriz de confusão, boxplots, etc.)
│   └── metrics/                # Métricas geradas (svm, árvore de decisão, etc.)
│
├── src/
│   ├── preprocessing.py        # Carga e limpeza dos dados
│   ├── feature_selection.py    # Seleção de características
│   ├── models.py               # Treinamento dos modelos
│   ├── evaluation.py           # Avaliação de performance
│   ├── utils.py                # Utilitários (salvar modelo, imprimir seções, etc.)
│
├── main.py                     # Script principal de execução
├── download_data.py            # Script de download do dataset
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## 📊 Conjunto de Dados

O dataset utilizado neste projeto está disponível no Kaggle:  
📎 [UCL | Matches & Players Data (2021/2022)](https://www.kaggle.com/datasets/azminetoushikwasi/ucl-202122-uefa-champions-league)

O arquivo `key_stats.csv` contém, entre outras, as seguintes colunas:

- `player_name`
- `club`
- `position` (target a ser previsto)
- `minutes_played`
- `match_played`
- `goals`
- `assists`
- `distance_covered`

## 🔍 Pipeline de Treinamento

1. 📥 Carregamento e limpeza dos dados (`src/preprocessing.py`)
2. ✂️ Seleção de características com `SelectKBest`
3. 🔀 Divisão entre treino e teste
4. 🤖 Treinamento com dois modelos:
   - Support Vector Machine (SVM)
   - Árvore de Decisão
5. 🧪 Avaliação com métricas:
   - Acurácia
   - Precisão
   - Recall
   - F1-score
6. 📊 Visualização com matrizes de confusão e boxplots

## 🚀 Como Executar

1. Clone o repositório:

```bash
   git clone https://github.com/seu-usuario/ucl_classification_project.git
   cd ucl_classification_project
```

2. Crie o ambiente virtual e instale as dependências:

```bash
   python -m venv .venv
   source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
   pip install -r requirements.txt
```

3. Baixe o dataset do projeto (se a pasta raw não estiver preenchida):
4. 
```bash
  python download_data.py
```

4. Execute o pipeline completo:

```bash
  python main.py
```

### 📓 Análise Exploratória (Opcional)

Para uma análise visual mais aprofundada das estatísticas dos jogadores, você pode executar o notebook Jupyter:

```bash
  jupyter notebook notebooks/exploracao_inicial.ipynb
```

Esse notebook gera gráficos como:

- Distribuição da variável alvo (`position`)
- Mapa de correlação entre atributos
- Boxplots comparando estatísticas por posição

> Obs: Os gráficos também são salvos em `results/plots/` para referência posterior.

## 🛠️ Tecnologias Usadas

- Python 3.10+
- pandas
- scikit-learn
- matplotlib
- seaborn (para visualizações complementares)

## 📈 Resultados Esperados

Ao final da execução, serão exibidas no console:

- Métricas de avaliação para cada modelo
- Matrizes de confusão
- Boxplots por posição

Além disso, os gráficos serão salvos em:

results/plots/

## 📄 Licença

MIT License © 2025 Robert Valadares
# ucl_classification_project
