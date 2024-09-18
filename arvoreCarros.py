import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Carregando os dados do arquivo ARFF
data, meta = arff.loadarff('carros.arff')

# Convertendo os dados categóricos em numéricos usando LabelEncoder
le_marca = LabelEncoder()
le_modelo = LabelEncoder()
le_resultado = LabelEncoder()

marca = le_marca.fit_transform(data['marca'])
modelo = le_modelo.fit_transform(data['modelo'])
ano = np.asarray(data['ano'], dtype=int).reshape(-1, 1)
km = np.asarray(data['quilometragem'], dtype=int).reshape(-1, 1)
valor = np.asarray(data['valor'], dtype=int).reshape(-1, 1)

# Definindo as features (apenas numericamente significativos)
features = np.concatenate((ano, km, valor), axis=1)

# Definindo o alvo (resultado: popular ou luxo)
target = le_resultado.fit_transform(data['resultado'])

# Criando e treinando a árvore de decisão
arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

# Convertendo as classes de numpy.bytes_ para str
class_names = [str(c.decode('utf-8')) for c in le_resultado.classes_]

# Plotando a árvore de decisão
plt.figure(figsize=(10, 10))
tree.plot_tree(arvore, feature_names=['Ano', 'Quilometragem', 'Valor'], 
               class_names=class_names, filled=True, rounded=True)
plt.show()

# Exibindo a matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(arvore, features, target,
                                      display_labels=class_names,
                                      values_format='d', ax=ax)
plt.show()
