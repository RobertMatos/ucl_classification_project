import pandas as pd
import os

def load_csv_data(file_path):
    """
    Carrega um arquivo CSV como DataFrame do pandas.

    Parâmetros:
        file_path (str): Caminho para o arquivo CSV.

    Retorna:
        pd.DataFrame: Dados carregados.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dados carregados com sucesso: {file_path}")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar os dados: {e}")
        raise
