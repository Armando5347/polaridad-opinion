import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("corpusNorm.csv", delimiter='\t')
print (data['Content'])

train_split, test_split = train_test_split(
        data,
        test_size=0.2,  # 20% para prueba y 80% para entrenamiento
        random_state=0,  # Semilla para asegurar reproducibilidad
        stratify=data['Polarity']  # Mantener proporciones de clase
    )

X_train = train_split["Content"]
y_train = train_split["Polarity"]
X_test = test_split["Content"]
y_test = test_split["Polarity"] 

clasificadores = [SVC(random_state=0), MLPClassifier(max_iter=1000, random_state=0)]
param_grid_svc = {
                'classifier__C': [0.1, 1, 10],  # Hiperparámetro C para SVM
                'classifier__kernel': ['linear', 'rbf', 'poly'],  # Tipo de kernel
                'classifier__gamma': ['scale', 'auto']  # Parámetro gamma
            }

param_grid_mlp = {
                'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'classifier__activation': ['tanh', 'relu'], #funcion de activacion
                'classifier__alpha': [0.0001, 0.001, 0.01]
            }
for clasificador in clasificadores:
  pipe = Pipeline([('text_representation', TfidfVectorizer(token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1,1))), ('classifier',clasificador)])
  #aqui, cv hace cross validation por su cuenta, y busca ajustar los mejores hipermarametros a partir del f1-macro
  if isinstance(clasificador, SVC):
    grid_search = GridSearchCV(pipe, param_grid_svc, cv=5,scoring='accuracy')
    print("Resultados de la maquina de soporte vectorial")
  else:
    grid_search = GridSearchCV(pipe, param_grid_mlp, cv=5, scoring='accuracy')
    print("Resultados del perceptrón multicapa")
  # Entrenar el modelo con GridSearchCV
  grid_search.fit(X_train, y_train)
  print(str(grid_search.best_params_))
  y_pred = grid_search.predict(X_test)
  print(classification_report(y_test, y_pred))
