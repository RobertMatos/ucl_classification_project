from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best(X, y, k=10):
    """
    Seleciona as k melhores features com base em testes ANOVA F-value.

    Parâmetros:
        - X: DataFrame com as features
        - y: vetor com os rótulos
        - k: número de features a serem selecionadas

    Retorna:
        - X_new: subconjunto de X com apenas as k melhores features
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Se quiser nomear quais colunas foram mantidas:
    mask = selector.get_support()
    selected_features = X.columns[mask]
    print("Features selecionadas:", list(selected_features))

    return X_new
