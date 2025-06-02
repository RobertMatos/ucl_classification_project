from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def train_svm(X_train, y_train):
    """
    Treina um classificador SVM com kernel linear.

    Retorna:
        - modelo treinado
    """
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    return svm


def train_decision_tree(X_train, y_train):
    """
    Treina um classificador Árvore de Decisão.

    Retorna:
        - modelo treinado
    """
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    return tree
