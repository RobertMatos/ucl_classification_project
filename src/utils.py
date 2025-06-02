import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - {model_name}")

    os.makedirs("results/plots", exist_ok=True)
    filename = f"results/plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

def save_model(model, filename):
    """
    Salva o modelo treinado em um arquivo .joblib
    """
    joblib.dump(model, filename)
    print(f"Modelo salvo em: {filename}")


def print_section(title):
    """
    Imprime um título de seção formatado no terminal.
    """
    print("\n" + "=" * 40)
    print(f"{title}")
    print("=" * 40)
