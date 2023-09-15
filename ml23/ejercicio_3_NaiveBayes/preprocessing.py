import numpy as np

def get_vocab(dataset):
    '''
        # Calcula un diccionario cuyas llaves sean las palabras de un vocabulario y el valor sea el indice de la palabra en el vocabulario.
        vocabulary = {
            'word_1': 0,
            'word_2': 1,
            'word_3': 2,
            ...
        }
    '''
    vocabulary={}
    idx = 0
    for sample in dataset:
        review = sample['text']
        words = review.split('')
        for w in words:
            if w not in vocabulary:
                vocabulary[w]=idx
                idx += 1
    vocabulary['UNK'] = idx
    return vocabulary

def get_one_hot_vector(text, vocabulary):
    ''' # TODO
        Dado un texto y un vocabulario, devuelve un vector one-hot
        donde el valor sea 1 si la palabra esta en el texto y 0 en caso contrario.
        Ejemplo:
            text = 'hola mundo'
            vocabulary = {
                'hola': 0,
                'mundo': 1,
                'UNK': 2
            }
            one_hot = [1, 1, 0]
    '''
    one_hot = np.zeros(len(vocabulary))
    # TODO: Genera el vector X dato el texto y vocabulario
    for i in range(len(text)):
        sample = text[i]
        for words in sample['text'].split():
            if words in vocabulary[words]:
                one_hot[words] = 1
    return one_hot

def preprocess_dataset(dataset, vocabulary):
    '''
        Datado un dataset (x, y) donde x es un texto y y es la etiqueta,
        devuelve una matriz X donde cada fila es un vector one-hot
        y un vector y con las etiquetas.
        Ejemplo:
            vocab = {"hola": 0, "mundo": 1, "alegre": 2}
            input: 
            dataset = [
                       {"text": "hola mundo alegre", "label": 0},
                       {"text": "hola mundo", "label": 1}
                       ]
            output:
            X = [[1, 1, 1],
                 [1, 1, 0]]
    '''
    X = []
    y = []
    for i in range(len(dataset)):
        sample = dataset[i]
        X.append(get_one_hot_vector(sample['text'], vocabulary))
        if 'label' in sample:
            y.append(sample['label'])
    return np.array(X), np.array(y)