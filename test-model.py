import numpy as np


def print_comps(conf_matrix):
    tn, fp, fn, tp = np.ravel(conf_matrix)
    print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

def get_accuracy(conf_matrix):
    tn, fp, fn, tp = np.ravel(conf_matrix)
    return (tp + tn)/(tp + fp + tn + fn)

def get_sensitivity(conf_matrix):
    tn, fp, fn, tp = np.ravel(conf_matrix)
    return (tp)/(tp + fn)

def get_specificity(conf_matrix):
    tn, fp, fn, tp = np.ravel(conf_matrix)
    return (tn)/(tn + fp)

def get_precision(conf_matrix):
    tn, fp, fn, tp = np.ravel(conf_matrix)
    return (tp)/(tp + fp)

def get_balanced_accuracy(conf_matrix):
    tn, fp, fn, tp = np.ravel(conf_matrix)
    return (get_specificity(conf_matrix) + get_sensitivity(conf_matrix))/2

def print_results(conf_matrix):
    print_comps(conf_matrix)
    print('Accuracy: {}'.format(get_accuracy(conf_matrix)))
    print('Sensitivity: {}'.format(get_sensitivity(conf_matrix)))
    print('Specificity: {}'.format(get_specificity(conf_matrix)))
    print('Precision: {}'.format(get_precision(conf_matrix)))
    print('BA: {}'.format(get_balanced_accuracy(conf_matrix)))
    
    
class Bruh():
    
    def __init__(self):
        self.mer = np.random.randint(0, 10)
        
    def get_mer(self):
        return self.mer

class Man():
    
    def __init__(self):
        self.mers = []
        for i in range(0, 10):
            self.mers.append(Bruh())
        
    def get_mer(self):
        return self.mer
    
    def get_mers(self):
        self.mers.sort(key=lambda x: x.get_mer(), reverse=True)

        
bleh = Man()
bleh.get_mers()

bark = []

dg = bark[0]


    


#a = np.matrix([[700, 0], [138, 0]])
#b = np.matrix([[0, 86], [0, 110]])
#c = np.matrix([[672, 28], [22, 116]])

#print_results(a)
#print_results(b)
#print_results(c)


