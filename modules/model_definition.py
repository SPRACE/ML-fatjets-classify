from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def model_definition():
    names = ['multilayer perceptron',
             'logistic regression',
             'random forest']

    classifiers = [
        MLPClassifier(solver='lbfgs',
                      hidden_layer_sizes=(5, ),
                      random_state=42),

        LogisticRegression(random_state=42),

        RandomForestClassifier(n_estimators=24,
                               max_depth=9,
                               random_state=42)]

    return names, classifiers
