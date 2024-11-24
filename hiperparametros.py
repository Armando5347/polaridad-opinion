import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import threading

data = pd.read_csv("corpusNorm.csv", delimiter='\t')
print (data['Content'])

train_split, test_split = train_test_split(
        data,
        test_size=0.2,  # 20% para prueba y 80% para entrenamiento
        random_state=0,  # Semilla para asegurar reproducibilidad
        stratify=data['Polarity']  # Mantener proporciones de clase
    )

X_train = train_split["Content"]
X_train_copy = X_train.copy()
y_train = train_split["Polarity"]
y_train_copy = y_train.copy()
X_test = test_split["Content"]
X_test_copy = X_test.copy()
y_test = test_split["Polarity"]
y_test_copy = y_test.copy()

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
def probarClasificador(clasificador, parametros, X_train, y_train, X_test, y_test, lock):
  pipe = Pipeline([('text_representation', TfidfVectorizer(token_pattern= r'(?u)\w+|\w+\n|\.|\¿|\?', ngram_range=(1,1))), ('classifier',clasificador)])
  #aqui, cv hace cross validation por su cuenta, y busca ajustar los mejores hipermarametros a partir del f1-macro
  grid_search = GridSearchCV(pipe, parametros, cv=5,scoring='f1_macro')
  if isinstance(clasificador, SVC):
    print("Resultados de la maquina de soporte vectorial")
  else:
    print("Resultados del perceptrón multicapa")
  # Entrenar el modelo con GridSearchCV
  grid_search.fit(X_train, y_train)
  y_pred = grid_search.predict(X_test)
  with lock:
    print(str(grid_search.best_params_))
    print(classification_report(y_test, y_pred))

lock = threading.Lock()

hilo_svc = threading.Thread(name="Experimento Maquina de soporte vectorial",target=probarClasificador, args=(clasificadores[0], param_grid_svc, X_train, y_train, X_test, y_test, lock))
hilo_mlp = threading.Thread(name="Experimento Perceptron multicapa",target=probarClasificador, args=(clasificadores[1], param_grid_mlp, X_train_copy, y_train_copy, X_test_copy, y_test_copy, lock))
#ejecutar hilos
hilo_svc.start()
hilo_mlp.start()
#esperar a que terminen
hilo_svc.join()
hilo_mlp.join()
