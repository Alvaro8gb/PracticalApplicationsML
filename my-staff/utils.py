from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    brier_score_loss
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from skrebate import ReliefF

from globals import SEED, TRAIN_SIZE, VALID_SIZE, TEST_SIZE, SCORE_METRIC, K_FOLD

from sklearn.preprocessing import StandardScaler


class ResultGridSearch(BaseModel):
    best_model: object
    best_num_features: int
    selected_feature_names: List[str]
    best_score: float


# Evaluaciones

def show_report(y_true, y_pred):
    report_str = classification_report(y_true, y_pred)
    report_dic = classification_report(y_true, y_pred, output_dict=True)

    print(report_dic)

    report_dic["accuracy"] = {'precision': " ", 'recall': " ", 'f1-score': report_dic["accuracy"],
                              'support': report_dic["macro avg"]["support"]}

    df = pd.DataFrame.from_dict(report_dic).transpose()

    index = list(df.index)

    index[1] = "Death"
    index[0] = "Censored"

    df.index = index

    df['support'] = df['support'].astype(int)

    latex_content = df.to_latex(float_format="%.2f", multicolumn_format="c")  # Format for multicolumns if needed

    latex_table = "\\begin{table}[H]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{ Report}\n"  # Reemplaza con el título deseado
    latex_table += "\\resizebox{0.95\columnwidth}{!}{\n"
    latex_table += latex_content + "}"
    latex_table += "\n\\end{table}"

    latex_table = latex_table.replace("accuracy", "\\hline \naccuracy")

    with open("../aux.tex", "w") as f:
        f.write(latex_table)

    print(report_str)


def show_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

    # Configurar etiquetas de los ejes
    plt.xlabel('Predicted Label')
    plt.ylabel('Real Label')

    # Mostrar el gráfico
    plt.show()


def split_valid_test(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    print("Len train val:", len(X_train_val), "Len test:", len(X_test))

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=VALID_SIZE / (TEST_SIZE + TRAIN_SIZE),
                                                      random_state=SEED)
    print("Len train:", len(X_train), "Len val:", len(X_val))

    return X_train, y_train, X_val, y_val, X_test, y_test


## GridSearch
def grid_search_no_fss(X, y, model, param_grid) -> ResultGridSearch:
    print("Starting evaluation with all variables")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORE_METRIC, cv=K_FOLD, n_jobs=-1)

    grid_search.fit(X, y)

    print(SCORE_METRIC, str(grid_search.best_score_))

    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    result = ResultGridSearch(best_model=best_model, best_num_features=len(X), selected_feature_names=list(X.columns),
                              best_score=best_score)
    return result


def grid_search_fss_score_func(X, y, model, param_grid, num_features: list[int],
                               score_func=mutual_info_classif) -> ResultGridSearch:
    best_score = -np.inf
    best_num_features = None
    selected_feature_names = None
    best_model = None

    print("Starting evaluation", str(num_features))

    for k in num_features:
        print("K", k)
        selector = SelectKBest(score_func=score_func, k=k)

        selector.fit_transform(X, y)
        selected_feature_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_feature_indices]
        print(selected_features)

        X_selected_univariant = X[selected_features]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORE_METRIC, cv=K_FOLD, n_jobs=-1)

        grid_search.fit(X_selected_univariant, y)

        print(k, SCORE_METRIC, str(grid_search.best_score_))

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_num_features = k
            best_model = grid_search.best_estimator_
            selected_feature_names = list(X_selected_univariant.columns)

    result = ResultGridSearch(best_model=best_model, best_num_features=best_num_features,
                              selected_feature_names=selected_feature_names, best_score=best_score)
    return result


def grid_search_fss_relieff(X, y, model, param_grid, num_features: list[int]) -> ResultGridSearch:
    best_score = -np.inf
    best_num_features = None
    best_model = None
    selected_feature_names = None

    for k in num_features:
        relieff_selector = ReliefF(n_features_to_select=k, n_neighbors=100)

        X_relieff = relieff_selector.fit_transform(X.values, y)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORE_METRIC, cv=K_FOLD, n_jobs=-1)

        grid_search.fit(X_relieff, y)

        print(k, SCORE_METRIC, str(grid_search.best_score_))

        selected_feature_indices = relieff_selector.top_features_[:k]
        selected_feature_names = list(X.columns[selected_feature_indices])

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_num_features = k
            best_model = grid_search.best_estimator_

    result = ResultGridSearch(best_model=best_model, best_num_features=best_num_features,
                              selected_feature_names=selected_feature_names, best_score=best_score)

    return result


def grid_search_fss_wrapper(X, y, model, param_grid, num_features: list[int]) -> ResultGridSearch:
    best_score = -np.inf
    best_num_features = None
    best_model = None
    selected_feature_names = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALID_SIZE, random_state=SEED)

    print("Train:", len(X_train), "Test:", len(X_test))

    for k in num_features:

        sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=k, direction="forward")

        sfs.fit(X_test, y_test)

        grid_search = GridSearchCV(model, param_grid, scoring=SCORE_METRIC, cv=K_FOLD)
        grid_search.fit(X_train, y_train)

        print(k, SCORE_METRIC, str(grid_search.best_score_))

        sfs_features = sfs.get_support()
        wrapper_features = X.columns[sfs_features]

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_num_features = k
            best_model = grid_search.best_estimator_
            selected_feature_names = list(wrapper_features)

    result = ResultGridSearch(best_model=best_model, best_num_features=best_num_features,
                              selected_feature_names=selected_feature_names, best_score=best_score)

    return result


def format_result(result: list):
    if result[0] is None:
        return "-"

    mean_value = np.mean(result)
    std_dev = np.std(result)

    latex_string = r"{:.4f} \pm {:.2f}".format(mean_value, std_dev)

    print(latex_string)

    return latex_string


def eval_model(y_true, y_pred, y_prob=None):
    brier_score = None
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="binary")

    print("Métricas de evaluación:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    if y_prob is not None:
        brier_score = brier_score_loss(y_true, y_prob)
        print(f"Brier Score: {brier_score:.4f}")

    return f1, brier_score


def pipline_evaluation(x: np.ndarray, y: np.ndarray, model, param_grid: dict, scoring=SCORE_METRIC):
    f1_score = []
    brie_score = []

    kf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=K_FOLD, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        print("-" * 40)
        print("GridSearchCV", SCORE_METRIC, str(grid_search.best_score_), str(grid_search.best_params_))
        best_model = grid_search.best_estimator_

        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)

        if hasattr(model, 'predict_proba'):
            y_proba = best_model.predict_proba(x_test)[:, 1]
        else:
            y_proba = None

        print("Train classes:", np.bincount(y_train))
        print("Test classes:", np.bincount(y_test))
        print("-" * 40)

        f1, brier = eval_model(y_test, y_pred, y_proba)

        f1_score.append(f1)

        brie_score.append(brier)

    return r"${}$ & ${}$ &".format(format_result(f1_score), format_result(brie_score))


def wrapper_ds(x: pd.DataFrame, y: np.ndarray, model, n_features):
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, scoring=SCORE_METRIC,
                                    direction="backward", cv=K_FOLD, n_jobs=-1)
    sfs.fit(x, y)
    print(x.columns[sfs.get_support()])

    x_selected_wrapper = sfs.transform(x)

    return x_selected_wrapper


## Plots
def show_data(X, y: np.ndarray):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)
    reduced_df = pd.DataFrame(data=reduced_features, columns=['X', 'Y'])

    # Crea un nuevo DataFrame que incluya las componentes principales y las clases predichas
    reduced_df['Predicted_Labels'] = y

    # Define una lista de colores para representar las clases
    palette = ['red', 'blue', 'green', 'yellow', 'orange']

    # Generar la lista de colores
    colors = [palette[i % len(palette)] for i in range(len(np.unique(y)))]

    # Crea un gráfico de dispersión con colores basados en las clases predichas
    for label, color in zip(reduced_df['Predicted_Labels'].unique(), colors):
        subset = reduced_df[reduced_df['Predicted_Labels'] == label]
        plt.scatter(subset['X'], subset['Y'], label=f'Clase {label}', color=color)

    plt.xlabel('Main Component 1')
    plt.ylabel('Main Component 2')
    plt.title('2D Visualization of Data with Predicted Classes (PCA)')
    plt.legend()
    plt.show()


def normalize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(X)
    scaled_df = pd.DataFrame(scaled_data, index=X.index, columns=X.columns)

    return scaled_df
