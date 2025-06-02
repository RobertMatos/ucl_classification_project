import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset bruto a partir de um CSV.
    """
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e prepara os dados do dataset UCL:
    - Remove colunas irrelevantes para o problema
    - Trata valores ausentes (mediana para numéricos, 'unknown' para categóricos)
    - Codifica variáveis categóricas (exceto a target 'position')
    """
    # Remover colunas que não serão usadas
    cols_to_drop = ['player_id', 'player_name', 'team_name', 'match_id', 'team_id']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Remover colunas com > 40% de valores nulos
    null_threshold = 0.4
    df = df.loc[:, df.isnull().mean() < null_threshold]

    # Preencher valores nulos
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('unknown')

    # Codificar variáveis categóricas (exceto target)
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'position':
            df[col] = LabelEncoder().fit_transform(df[col])

    # Filtrar linhas com target ausente
    df = df[df['position'].notna()]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/key_stats_clean.csv", index=False)

    return df

def get_features_and_labels(df: pd.DataFrame):
    """
    Separa features (X) e target (y).
    Target aqui é a coluna 'position'.
    """
    X = df.drop(columns=['position'])
    y = df['position']
    return X, y
