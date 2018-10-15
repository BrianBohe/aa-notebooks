# decision_tree/decision_tree.py
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

# Definición de la estructura del árbol. 
class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
    def __init__(self, etiquetas):
        etiquetas = np.array(etiquetas[0])
        self.cuentas = dict(Counter(etiquetas))
    def es_hoja(self):
        return True

class Nodo_De_Decision:
    # Un Nodo de Decisión contiene preguntas y una referencia al sub-árbol izquierdo y al sub-árbol derecho
     
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho
    
    def es_hoja(self):
        return False
        
# Definición de la clase "Pregunta"
class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor
    
    def cumple(self, instancia):
        # Devuelve verdadero si la instancia cumple con la pregunta
        return instancia[self.atributo] >= self.valor
    
    def __repr__(self):
        return "¿Es el valor para {} mayor o igual a {}?".format(self.atributo, self.valor)

def gini(etiquetas):
    # COMPLETAR
    total = len(etiquetas)
    et = etiquetas.groupby(0).size().reset_index(name='counts')
    et['step'] = (et['counts']/total)**2

    return 1 - et['step'].values.sum()

def ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    # COMPLETAR
    etiquetas_padre = etiquetas_rama_izquierda.append(etiquetas_rama_derecha)
    gini_padre = gini(etiquetas_padre)
    gini_izq = gini(etiquetas_rama_izquierda)
    gini_der = gini(etiquetas_rama_derecha)
    ganancia_gini = gini_padre - (gini_izq*len(etiquetas_rama_izquierda)/len(etiquetas_padre) + gini_der*len(etiquetas_rama_derecha)/len(etiquetas_padre))
    return ganancia_gini


def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    # COMPLETAR (recomendamos utilizar máscaras para este punto)
    mask = pregunta.cumple(instancias)
    instancias_cumplen = instancias[mask]
    etiquetas_cumplen = etiquetas[mask]
    instancias_no_cumplen = instancias[~mask]
    etiquetas_no_cumplen = etiquetas[~mask]
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen

def encontrar_mejor_atributo_y_corte(instancias, etiquetas, params):
    max_ganancia = 0
    mejor_pregunta = None
    columnas_shuffled = instancias.columns.tolist()
    shuffle(columnas_shuffled)
    for columna in columnas_shuffled[:params['max_attr']]:
        for corte in np.linspace(instancias[columna].min(), instancias[columna].max(), params['grid_space']):
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, corte)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
   
            ganancia = ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
            
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta
    return max_ganancia, mejor_pregunta


def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")


def construir_arbol(instancias, etiquetas, depth, params):
    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas, params)
    
    print(depth)
    # Criterio de corte: ¿Hay ganancia?
    if ganancia == 0 or depth == 1:
        #  Si no hay ganancia en separar, no separamos. 
        return Hoja(etiquetas)
    else: 
        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen, depth - 1, params)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, depth - 1, params)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
        
        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)


def predecir(arbol, x_t):
    if arbol.es_hoja():
        return arbol.cuentas

    if arbol.pregunta.cumple(x_t):
        return predecir(arbol.sub_arbol_izquierdo, x_t)
    else:   
        return predecir(arbol.sub_arbol_derecho, x_t)

class MiClasificadorArbol(): 
    def __init__(self, max_depth = 3, grid_space = 10, max_attr = 20):
        self.arbol = None
        self.params = {'max_depth': max_depth, 'grid_space': grid_space, 'max_attr': 20}

    def get_params(self, deep = True):
        return self.params

    def fit(self, X_train, y_train):
        self.arbol = construir_arbol(pd.DataFrame(X_train), pd.DataFrame(y_train), self.params['max_depth'], self.params)
        self.classes = set()
        for e in y_train:
            self.classes.add(e)

        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t])
            x_t_df.columns = x_t_df.columns.astype(str)
            x_t_df = x_t_df.iloc[0]
            prediction = predecir(self.arbol, x_t_df)

            maxCount = 0
            maxKey = ""
            for k,v in arbol.cuentas.items():
                if v > maxCount:
                    maxCount = v
                    maxKey = k
            predictions.append(maxKey)

        return predictions

    def predict_proba(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t])
            x_t_df.columns = x_t_df.columns.astype(str)
            x_t_df = x_t_df.iloc[0]
            prediction = predecir(self.arbol, x_t_df)
            total_instances = 0
            for k,v in prediction.items():
                total_instances += v

            probabilities = [0 for i in range(len(self.classes))]
            for e in self.classes:
                if prediction[e] is not None:
                    probabilities[e] = prediction[e]/total_instances
            predictions.append(probabilities)
        return np.asarray(predictions)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy