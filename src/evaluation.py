from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_models(models, model_names, X_test, y_test):
    os.makedirs("results/metrics", exist_ok=True)

    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"\n📊 Avaliação do modelo: {name}")
        print(f"Acurácia:  {acc:.4f}")
        print(f"Precisão:  {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")

        with open(f"results/metrics/{name.lower().replace(' ', '_')}_metrics.txt", "w") as f:
            f.write(f"Acurácia:  {acc:.4f}\n")
            f.write(f"Precisão:  {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-score:  {f1:.4f}\n")
