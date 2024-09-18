import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


data,meta = arff.loadarff('./CriterioProvas.arff')
attributes = meta.names()
data_value = np.asarray(data)
percFalta = np.asarray(data['PercFalta']).reshape(-1,1)
notas = np.asarray(data['P1']).reshape(-1,1)
notas2 = np.asarray(data['P2']).reshape(-1,1)
features = np.concatenate((notas, notas2, percFalta),axis=1)

target = data['resultado']
Arvore = DecisionTreeClassifier(criterion='entropy').fit(features,
target)
plt.figure(figsize=(10, 10))
tree.plot_tree(Arvore,feature_names=['P1', 'P2',
'PercFalta'],class_names=['Aprovado', 'Reprovado'],filled=True, rounded=True)
plt.show()
fig, ax = plt.subplots(figsize=(25, 10))
ConfusionMatrixDisplay.from_estimator(Arvore, features, target,
                                      display_labels=['Aprovado', 'Reprovado'],
                                      values_format='d', ax=ax)
plt.show()