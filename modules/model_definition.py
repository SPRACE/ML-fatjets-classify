from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def model_definition():
    names = ['multilayer perceptron',
             'random forest',
             'logistic regression']

    classifiers = [
        MLPClassifier(solver='lbfgs',
                      hidden_layer_sizes=(5, ),
                      random_state=1),

        RandomForestClassifier(n_estimators=24,
                               max_depth=9,
                               random_state=1),

        LogisticRegression(random_state=1)]

    return names, classifiers
