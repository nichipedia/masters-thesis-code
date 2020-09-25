import numpy as np
import numbers
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import cross_val_score


class Individual():
    """
    Member of the GeneticPopulation
    """
    def __init__(self, estimator, X, y, one_hots):
        df = pd.DataFrame(data=X, index=None, columns=None)
        new_X = df[df.columns[one_hots.astype(bool)]].values
        self.score = cross_val_score(estimator, new_X, y, cv=10).mean()
        self.one_hots = one_hots

    def get_score(self):
        return self.score

    def get_one_hots(self):
        return self.one_hots

    def mutate(self):
        num_features = len(self.one_hots)
        mutation = np.random.randint(0, num_features)
        self.one_hots[mutation] = np.abs(self.one_hots[mutation]-1)


class GeneticPopulation():
    """
    The gene pool
    """


    def __init__(self, estimator, X, y):
        self.num_features = X.shape[1]
        self.population = []
        self.X = X
        self.y = y
        self.estimator = estimator
        self.census = 10*self.num_features
        for i in range(0, self.census):
            self.one_hots = np.random.randint(0, 2, size=self.num_features)
            self.population.append(Individual(self.estimator, self.X, self.y, self.one_hots))

    def evaluate(self):
        self.population.sort(key=lambda x: x.get_score(), reverse=True)
        return self.population[0].get_score()

    def selection(self):
        temp = []
        top = int(0.01 * self.census)
        temp.append(self.population[:top])
        self.population.remove(self.population[:top])
        roulette = 0
        for individual in self.population:
            roulette = roulette + individual.get_score()

        for i in range(0, self.census/2):
            selection = np.random.randint(0,roulette)
            spin = 0
            for individual in self.population:
                spin = spin + individual.get_score()
                if (spin > roulette):
                    temp.append(individual)
                    self.population.remove(individual)
                    roulette = roulette - individual.get_score()
                    break

        self.population = temp

    def crossover(self):
        temp = []
        while(len(self.population) > 0 or len(self.population) == 1):
            count = len(self.population)
            father_idx = np.random.randint(0,count)
            mother_idx = np.random.randint(0,count-1)
            father = self.population.pop(obj=father_idx)
            mother = self.population.pop(obj=mother_idx)
            father_dna = father.get_one_hots()
            mother_dna = mother.get_one_hots()
            favorite_child_dna = np.zeros(self.num_features)
            hated_child_dna = np.zeros(self.num_features)
            for i in range(self.num_features):
                token = np.random.randint(0, 100)
                if (token > 49):
                    favorite_child_dna[i] = father_dna[i]
                    hated_child_dna[i] = mother_dna[i]
                else:
                    favorite_child_dna[i] = mother_dna[i]
                    hated_child_dna[i] = father_dna[i]
            favorite_child = Individual(self.estimator, self.X, self.y, favorite_child_dna)
            hated_child = Individual(self.estimator, self.X, self.y, hated_child_dna)
            temp.append(favorite_child)
            temp.append(hated_child)
            temp.append(father)
            temp.append(mother)
        return temp


    def mutation(self):
        for individual in self.population:
            token = np.random.randint(0, 100)
            if (token < 2):
                individual.mutate()

    def get_alpha(self):
        return self.population[0].get_one_hots()




class FeatureGA(BaseEstimator, MetaEstimatorMixin):
    """
    Genetic algorithm for feature selection.
    """

    def __init__(self, fittness, epochs=1000, threshold=None, prefit=False, norm_order=1, max_features=1000):
        self.fittness = fittness
        self.epochs = epochs
        self.threshold = None
        self.prefit = False
        self.norm_order = 1
        self.max_features = max_features
        self.alpha = None

    def fit(self, X, y=None, **fit_params):
        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError("'max_features' should be an integer between"
                    " 0 and {} features. Got {!r} instead."
                    .format(X.shape[1], self.max_features))
        elif self.max_features < 0 or self.max_features > X.shape[1]:
            raise ValueError("'max_features' should be 0 and {} features."
                "Got {} instead.".format(X.shape[1], self.max_features))

        if self.prefit:
            raise NotFittedError("Since 'prefit=True', call transform directly")
        num_features = X.shape[1]
        world = GeneticPopulation(self.fittness, X, y)

        epoch = 0
        while (world.evaluate() < 0.85 or epoch < self.epochs):
            print('Epoch: {} ---- Current Highest Fittness: {}'.format(epoch, world.evaluate()))
            world.selection()
            world.crossover()
            world.mutation()

        self.alpha = world.get_alpha()
        return self

    def transform(self, X):
        """

        """
        print(self.alpha)

