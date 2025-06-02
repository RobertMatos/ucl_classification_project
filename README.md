# UCL Player Position Classifier ⚽️

This project applies machine learning techniques to predict UEFA Champions League player positions (e.g., defender, midfielder, forward) based on key statistics such as goals, assists, and minutes played.

## 📁 Project Structure

    ucl_classification_project/
    │
    ├── data/
    │   ├── raw/
    │   │   └── key_stats.csv       # Original player data
    │   └── processed/              # Cleaned data
    │
    ├── results/
    │   └── plots/                  # Generated charts (confusion matrix, boxplots, etc.)
    │   └── metrics/                # Stored metrics (SVM, decision tree, etc.)
    │
    ├── src/
    │   ├── preprocessing.py        # Data loading and cleaning
    │   ├── feature_selection.py    # Feature selection
    │   ├── models.py               # Model training
    │   ├── evaluation.py           # Performance evaluation
    │   ├── utils.py                # Utilities (save model, section printing, etc.)
    │
    ├── main.py                     # Main pipeline script
    ├── download_data.py            # Dataset download script
    ├── requirements.txt            # Project dependencies
    └── README.md                   # This file

## 📊 Dataset

The dataset used in this project is available on Kaggle:  
📎 UCL | Matches & Players Data (2021/2022)  
https://www.kaggle.com/datasets/azminetoushikwasi/ucl-202122-uefa-champions-league

The key_stats.csv file includes the following relevant columns:

- player_name
- club
- position (target to be predicted)
- minutes_played
- match_played
- goals
- assists
- distance_covered

## 🔍 Training Pipeline

1. 📥 Load and clean data (src/preprocessing.py)
2. ✂️ Feature selection using SelectKBest
3. 🔀 Split into training and test sets
4. 🤖 Train two models:
   - Support Vector Machine (SVM)
   - Decision Tree Classifier
5. 🧪 Evaluate with the following metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
6. 📊 Visualize with confusion matrices and boxplots

## 🚀 How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/ucl_classification_project.git
    cd ucl_classification_project
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

3. Download the dataset (if data/raw is empty):

    ```bash
    python download_data.py
    ```

4. Run the pipeline:

    ```bash
    python main.py
    ```

### 📓 Exploratory Analysis (Optional)

To visually explore player statistics, run the following notebook:

    ```bash
    jupyter notebook notebooks/exploracao_inicial.ipynb
    ```

This notebook generates plots such as:

- Distribution of the target variable (position)
- Correlation heatmap between numerical features
- Boxplots comparing statistics by position

> Note: The plots are also saved to results/plots/ for later reference.

## 🛠️ Technologies Used

- Python 3.10+
- pandas
- scikit-learn
- matplotlib
- seaborn (optional, for extra visualizations)

## 📈 Expected Results

After execution, you’ll see:

- Evaluation metrics for each model
- Confusion matrices
- Boxplots per position

All generated plots will be saved to:

    results/plots/

## 📄 License

MIT License © 2025 Robert Valadares
