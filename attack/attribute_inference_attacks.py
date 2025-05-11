"""
Author: Yudi Xiong
Google Scholar: https://scholar.google.com/citations?user=LY4PK9EAAAAJ
ORCID: https://orcid.org/0009-0001-3005-8225
Date: April, 2024
Example command to run:
python attribute_inference_attacks.py --test_size=0.7 --csv_file_path='user_camouflaged_embeddings_gender_128_0.55.csv'
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import argparse
from collections import Counter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Run attribute_inference_attacks.")
    parser.add_argument('--test_size', type=float, default=0.7) # attacker train_size = 1 - test_size = 0.3 = 30%
    parser.add_argument('--csv_file_path', nargs='?', default='user_camouflaged_embeddings_gender_128_0.4.csv',
                        help='The goal and path of attribute inference attacks (using embeddings).')
    return parser.parse_args()
args = parse_args()

np.random.seed(2024)
n = 2068
d = 128
X = np.random.rand(n, d) - 0.5

file_path = args.csv_file_path

df = pd.read_csv(file_path, header=None)

user_item_matrix = df.values

file_path = '../Data/QBarticle_QBvideo/new_reindex.txt'

df = pd.read_csv(file_path, sep='\t', header=None)


df.columns = ['User_ID', 'Item_ID', 'Rating', 'Gender', 'Age']


user_gender_matrix = df[['User_ID', 'Gender']].drop_duplicates().reset_index(drop=True)

user_gender_matrix['Gender'] = user_gender_matrix['Gender'] - 1

user_age_matrix = df[['User_ID', 'Age']].drop_duplicates().reset_index(drop=True)

user_age_matrix['Age'] = user_age_matrix['Age'] - 1

Y_gender = user_gender_matrix.iloc[:, 1].tolist()
Y_age = user_age_matrix.iloc[:, 1].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, Y_gender, test_size=args.test_size, random_state=0) # XGBoost random_state=1,others = 0
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(user_item_matrix, Y_gender, test_size=args.test_size, random_state=0)

models = {
    "SVM": SVC(C=1, kernel='linear'),
    'GBDT': GradientBoostingClassifier(n_estimators=300, max_depth=3),
    "Logistic Regression": LogisticRegression(C=1, max_iter=200),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1),
}

results_accuracy_random = {}
results_f1_random = {}
accuracies_random = []
f1_scores_random = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results_accuracy_random[name] = accuracy
    results_f1_random[name] = f1
    accuracies_random.append(accuracy)
    f1_scores_random.append(f1)
average_accuracy_random = np.mean(accuracies_random)
average_f1_score_random = np.mean(f1_scores_random)
formatted_results_accuracy_random = {k: f"{v:.4f}" for k, v in results_accuracy_random.items()}
formatted_results_f1_random = {k: f"{v:.4f}" for k, v in results_f1_random.items()}
print("Random inference accuracy:", formatted_results_accuracy_random)
print("Random inference f1:", formatted_results_f1_random)


results_accuracy_true = {}
results_f1_true = {}
accuracies_true = []
f1_scores_true = []
for name, model in models.items():
    model.fit(X_train_2, y_train_2)
    y_pred_2 = model.predict(X_test_2)
    accuracy = accuracy_score(y_test_2, y_pred_2)
    f1 = f1_score(y_test_2, y_pred_2, average='weighted')
    results_accuracy_true[name] = accuracy
    results_f1_true[name] = f1
    accuracies_true.append(accuracy)
    f1_scores_true.append(f1)
average_accuracy_true = np.mean(accuracies_true)
average_f1_score_true = np.mean(f1_scores_true)
formatted_results_accuracy_true = {k: f"{v:.4f}" for k, v in results_accuracy_true.items()}
formatted_results_f1_true = {k: f"{v:.4f}" for k, v in results_f1_true.items()}
print("Inference accuracy:", formatted_results_accuracy_true)
print("Inference f1:", formatted_results_f1_true)
