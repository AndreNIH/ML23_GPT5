import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha

    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
        total=0
        sum=0
        for i in c<=len(y):
                if y[total]==1:
                    sum=sum+1 
        self.prior_positive=sum/total
        
        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = 1-self.prior_positive

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
        
        self._likelihoods_positives = [len(y)]
        self._likelihoods_negatives = [len(y)]
        #np.sum(...)
        i=0
        for i in i<=len(y):
            if y[i]==1:
                self._likelihoods_positives [i] = (np.sum(X[i], axis=0))/sum
            else:
                self._likelihoods_negatives [i] = (np.sum(X[i], axis=0))/(total-sum)
        #P(y|X)=(P(X|y)*P(y))/P(X)
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]
        return self

    def predict(self, X):
        # TODO: Calcula la distribución posterior para la clase 1 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 1 | X) = P(y=1) * P(x1|y=1) * P(x2|y=1) * ... * P(xn|y=1)
        mult=self.prior_positive
        i=0
        for i in self._likelihoods_positives:
            if(self._likelihoods_positives[i]>0):
                mult = mult * (self._likelihoods_positives[i])
            else:
                mult = mult * (1-self._likelihoods_negatives[i])
        posterior_positive = mult

        # TODO: Calcula la distribución posterior para la clase 0 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 0 | X) = P(y=0) * P(x1|y=0) * P(x2|y=0) * ... * P(xn|y=0)
        mult=self.prior_negative
        i=0
        for i in self._likelihoods_negatives:
            mult = mult * (self._likelihoods_negatives[i])
        posterior_negative = mult 

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        clase = 
        return clase
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)