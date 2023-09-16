import numpy as np
import math

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha
        self._likelihoods_negatives = []
        self._likelihoods_positives = []

    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
<<<<<<< HEAD
        self.prior_positive = sum(n for n in y)/len(y)
        
        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = 1-self.prior_positive
=======
        # self.prior_positives = 

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        # self.prior_negative = 
>>>>>>> upstream/master

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
<<<<<<< HEAD

        #P(y|X)=(P(X|y)*P(y))/P(X)
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]
                
        #Esto tiene que inicializarse con unos, de otra forma
        #no podemos utilizar logaritmos para evitar underflows
        self._likelihoods_negatives = np.ones(len(X[0]))
        self._likelihoods_positives = np.ones(len(X[0]))
        positiveWords = sum(len(X[sentenceIndex] * y[sentenceIndex] == 1) for sentenceIndex in range(0,len(X)))
        negativeWords = sum(len(X[sentenceIndex] * y[sentenceIndex] == 0) for sentenceIndex in range(0,len(X)))
        
        for i in range(0,len(y)):
            sentence = X[i]
            if y[i] == 1:
                self._likelihoods_positives += sentence
            else:
                self._likelihoods_negatives += sentence
        self._likelihoods_positives /= positiveWords
        self._likelihoods_negatives /= negativeWords
=======
        # self._likelihoods_positives = 
        
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]

        # self._likelihoods_negatives = _likelihoods_negatives
>>>>>>> upstream/master
        return self

    def predict(self, X):
        # TODO: Calcula la distribución posterior para la clase 1 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 1 | X) = P(y=1) * P(x1|y=1) * P(x2|y=1) * ... * P(xn|y=1)
<<<<<<< HEAD
        

        pred = []
        for sentence in X: sentence = [n+self.alpha for n in sentence]
        for sentence in X:
            pLogSum = math.log(self.prior_positive)
            nLogSum = math.log(self.prior_negative)
            for wordIndex in range(0, len(sentence)):
                if sentence[wordIndex] == 1:
                    pLogSum += math.log(self._likelihoods_positives[wordIndex])
                    nLogSum += math.log(self._likelihoods_negatives[wordIndex])

            pred.append(pLogSum > nLogSum)  # Clasificacion

        return pred

        
        #Unoptimized, underflowed code
        '''
        for sentence in X:
            pMult = self.prior_positive
            nMult = self.prior_negative

            for wordIndex in range(0,len(sentence)):
                if sentence[wordIndex]:
                    pMult *= self._likelihoods_positives[wordIndex]
                    nMult *= self._likelihoods_negatives[wordIndex]

            

            pred.append(pMult > nMult) #Clasificacion
        return pred
        '''
=======
        # posterior_positive = 

        # TODO: Calcula la distribución posterior para la clase 0 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 0 | X) = P(y=0) * P(x1|y=0) * P(x2|y=0) * ... * P(xn|y=0)
        # posterior_negative = 

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        # clase = 
        return
>>>>>>> upstream/master
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)