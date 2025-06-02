import kagglehub
from pathlib import Path
import shutil

def download_ucl_dataset():
    print("Baixando base da UCL via KaggleHub...")
    path = kagglehub.dataset_download("azminetoushikwasi/ucl-202122-uefa-champions-league")

    print(f"Base baixada em: {path}")

    # Define destino: ./data/raw/
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Move todos os arquivos baixados para ./data/raw/
    for file in Path(path).glob("**/*"):
        if file.is_file():
            shutil.copy(file, target_dir / file.name)
            print(f"Copiado: {file.name}")

    print("Download e cópia concluídos.")

if __name__ == "__main__":
    download_ucl_dataset()
