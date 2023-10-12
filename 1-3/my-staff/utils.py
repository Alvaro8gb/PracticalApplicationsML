import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from globals import PATH_DS, SEED, TRAIN_SIZE, VALID_SIZE, TEST_SIZE


# PACKING Y UNPACKING

def pack(data, target, features_names, class_names):
    return dict(data=data, target=target, features_names=features_names, class_names=class_names)


def unpack(ds):
    return ds["data"], ds["target"], ds["features_names"], ds["class_names"]


def load_ds(name: str):
    full = pd.read_pickle(PATH_DS + name + ".pkl")
    X, y, features_names, class_names = unpack(full)
    return X, y, features_names, class_names


def dump_ds(name: str, X, y, features_names, class_names):
    full = pack(X, y, features_names, class_names)
    path = PATH_DS + name + ".pkl"
    pd.to_pickle(full, path)
    print("Saved in ", path)


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

    latex_content = df.to_latex(
        float_format="%.2f",
        multicolumn_format="c")  # Format for multicolumns if needed

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


def eval_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Métricas de evaluación:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


def split_valid_test(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                                y,
                                                                test_size=TEST_SIZE,
                                                                random_state=SEED)

    print("Len train val:", len(X_train_val), "Len test:", len(X_test))

    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                      y_train_val,
                                                      test_size=VALID_SIZE / (TEST_SIZE + TRAIN_SIZE),
                                                      random_state=SEED)
    print("Len train:", len(X_train), "Len val:", len(X_val))

    return X_train, y_train, X_val, y_val, X_test, y_test


def show_data(X, y):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)
    reduced_df = pd.DataFrame(data=reduced_features, columns=['X', 'Y'])

    # Crea un nuevo DataFrame que incluya las componentes principales y las clases predichas
    reduced_df['Predicted_Labels'] = y

    # Define una lista de colores para representar las clases
    palette = ['red', 'blue', 'green', 'yellow', 'orange']

    # Generar la lista de colores
    colors = [palette[i % len(palette)] for i in range(len(y.unique()))]

    # Crea un gráfico de dispersión con colores basados en las clases predichas
    for label, color in zip(reduced_df['Predicted_Labels'].unique(), colors):
        subset = reduced_df[reduced_df['Predicted_Labels'] == label]
        plt.scatter(subset['X'], subset['Y'], label=f'Clase {label}', color=color)

    plt.xlabel('Main Component 1')
    plt.ylabel('Main Component 2')
    plt.title('2D Visualization of Data with Predicted Classes (PCA)')
    plt.legend()
    plt.show()
