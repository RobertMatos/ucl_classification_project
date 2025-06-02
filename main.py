# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, clean_data
from src.feature_selection import select_k_best
from src.models import train_svm, train_decision_tree
from src.evaluation import evaluate_models
from src.utils import plot_confusion_matrix, print_section

def main():
    # 1. Carregar e preparar os dados
    print_section("1. Carregando e pré-processando os dados")
    df_raw = load_data("data/raw/key_stats.csv")
    df = clean_data(df_raw)

    # Suponha que 'position' seja a variável a ser prevista
    X = df.drop(columns=["position"])  # ou outro target escolhido
    y = df["position"]

    # 2. Selecionar features
    print_section("2. Seleção de características")
    X_selected = select_k_best(X, y, k=10)

    # 3. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # 4. Treinamento dos modelos
    print_section("3. Treinando modelos")
    svm_model = train_svm(X_train, y_train)
    tree_model = train_decision_tree(X_train, y_train)

    # 5. Avaliação
    print_section("4. Avaliação dos modelos")
    evaluate_models(
        models=[svm_model, tree_model],
        model_names=["SVM", "Árvore de Decisão"],
        X_test=X_test,
        y_test=y_test
    )

    # 6. Matriz de confusão
    print_section("5. Matrizes de confusão")
    for model, name in zip([svm_model, tree_model], ["SVM", "Árvore de Decisão"]):
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name=name)

if __name__ == "__main__":
    main()
