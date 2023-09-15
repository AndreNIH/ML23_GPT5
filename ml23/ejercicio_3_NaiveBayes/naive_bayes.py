import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha
        self._likelihoods_negatives = []
        self._likelihoods_positives = []

    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
        psum=0
        for i in y:
            if y[i]==1:
                psum=psum+1 
        self.prior_positive=psum/len(y)
        
        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = 1-self.prior_positive

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
        
        for review in X:
            self._likelihoods_negatives.append(np.ones(len(review)))
            self._likelihoods_positives.append(np.ones(len(review)))
            #self._likelihoods_positives = np.ones(len(X[0]))
            #self._likelihoods_negatives = np.ones(len(X[0]))
        #np.sum(...)
        i=0
        ia=0
        ib=0
        '''
        for i in y:
            if y[i]==1:
                self._likelihoods_positives [ia] = (np.sum(X[i], axis=0))/sum
                ia=+1
            else:
                self._likelihoods_negatives [ib] = (np.sum(X[i], axis=0))/(total-sum)
                ib=+1
        '''

        
        
        totalPositiveReviews = sum(reviewType == 1 for reviewType in y)
        totalNegativeReviews = sum(reviewType == 0 for reviewType in y)
        for i in range(0, len(y)):
            #Branch positivo
            if y[i] == 1:
                for word in X[i]:
                    occurancesInReview = (X[i] == word).sum() #Esto esta roto
                    print(occurancesInReview)

        
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
        cont=0
        for i in X:
            for j in i:
                if(j==1):
                    mult = mult * (self._likelihoods_positives[cont])
                else:
                    mult = mult * (1-self._likelihoods_negatives[cont])
                cont=+1   
        posterior_positive = mult

        # TODO: Calcula la distribución posterior para la clase 0 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 0 | X) = P(y=0) * P(x1|y=0) * P(x2|y=0) * ... * P(xn|y=0)
        mult=self.prior_negative
        cont=0
        for i in X:
            for j in i:
                if(j==1):
                    mult = mult * (self._likelihoods_negatives[cont])
                else:
                    mult = mult * (1-self._likelihoods_positives[cont])
                cont=+1   
        posterior_negative = mult

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        if ((posterior_positive*self.prior_positive)>(posterior_negative*self.prior_negative)):
            clase=1 
        else:
            clase=0
        return clase
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)