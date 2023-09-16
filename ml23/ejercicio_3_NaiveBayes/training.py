from datasets import load_dataset
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from preprocessing import get_vocab, preprocess_dataset
from naive_bayes import NaiveBayes
THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(THIS_PATH, 'rotten_tomatoes_dataset.py')

'''
    Instrucciones:
    - En preprocessing.py completa el método get_one_hot_vector
    - En naive_bayes.py completa los métodos fit y predict
    - cuando finalices corre este archivo para entrenar y evaluar tu modelo.

    Entrega en blackboard a modo texto:
     - Dos ejemplos falsos positivos
     - Dos ejemplos de falsos negativos
     - Dos ejemplos de verdaderos positivos
     Para el conjunto de validación.
'''


def print_samples(dataset, n_samples, random=True):
    if random:
        indices = np.random.randint(0, len(dataset), n_samples)
    else:
        indices = np.arange(n_samples)

    for i in indices:
        idx = i.item()
        datapoint = dataset[idx]
        text = datapoint['text']
        label = datapoint['label']
        is_pos = "positive" if label else "negative"
        print(f"({is_pos}) - Text: {text}")

if __name__ == "__main__":
    # Carga de datos
    dataset = load_dataset(DATASET_PATH)
    training_set = dataset['train']
    validation_set = dataset['validation']
    test_set = dataset['test']

    print_samples(training_set, 5)

    # Preprocesamiento
    vocabulary = get_vocab(training_set)
    X_train, y_train = preprocess_dataset(training_set, vocabulary)
    X_val, y_val = preprocess_dataset(validation_set, vocabulary)

    # Entrenamiento
    # Sklearn
    alpha = 1
    sk_model = MultinomialNB(alpha = alpha)
    sk_model.fit(X_train, y_train)
    pred = sk_model.predict(X_val)
    sk_score = np.sum(pred == y_val) / len(y_val)
    
    # Propio
    model = NaiveBayes(alpha)
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    #Prediccion
    ejemplos=dataset['validation'][:2]
    #obtener representacion de vector de mis ejemplos
    ejemplos_x, ejemplos_label = preprocess_dataset(dataset['validation'], vocabulary)
    predicciones=model.predict(ejemplos_x[:2])    
    for ej,pred in zip(ejemplos['text'], predicciones):
         print(f'\nText={ej}.\nPrediction:{"Positiva" if pred else "Negativa"}\n')
    
    # Evaluación
    print(f"sk-learn accuracy: {sk_score} \t Propio accuracy: {accuracy}")