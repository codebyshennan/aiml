from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

# define the hyperparameter grid for the models
logistic_params = {
    "penalty": ["l2", "l2", "elasticnet"],
    "C": [0.1, 1, 10],
    "solver": ["newton-cg", "lbfgs", "liblinear"],
    "multi_class": ["multinomial"],
    "max_iter": [400],
}

decision_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 20, 30, 40, 50],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}

forest_params = {
    "n_estimators": [10, 20, 30, 40, 50],
    "max_depth": [5, 10, 20, 30, 40, 50],
    "min_sample_leafs": [1, 2, 3, 4, 5],
}

bernoulli_nb_params = {"alpha": [0.1, 0.5, 1.0], "fit_prior": [True, False]}

mlp_params = {
    "hidden_layer_sizes": [(50, 50, 50), (100, 100, 100), (100, 100)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam"],
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": [0.001, 0.01, 0.1, 1],
    "alpha": [0.0001, 0.001],
    "max_iter": [400],
}

svc_params = {
    "C": [0.1, 1, 10],
    "loss": ["hinge", "squared_hinge"],
    "multi_class": ["ovr"],
    "max_iter": [800],
}

knn_params = {
    "n_neighbors": [2, 5, 10, 20],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}


class Models:
    def __init__(self):
        self.registry = [
            {
                "name": "Logistic Regression",
                "model": LogisticRegression(),
                "params": logistic_params,
            },
            {
                "name": "Decision Tree",
                "model": DecisionTreeClassifier(),
                "params": decision_params,
            },
            {
                "name": "Random Forest",
                "model": RandomForestClassifier(),
                "params": decision_params,
            },
            {
                "name": "Naive Bayes, Bernoulli",
                "model": BernoulliNB(),
                "params": bernoulli_nb_params,
            },
            {
                "name": "Linear Support Vector Classification",
                "model": LinearSVC(),
                "params": svc_params,
            },
            {
                "name": "K-Nearest Neighbors",
                "model": KNeighborsClassifier(),
                "params": knn_params,
            },
            {
                "name": "Multi-layered Perceptron",
                "model": MLPClassifier(),
                "params": mlp_params,
            },
        ]

    def addNewModel(self, model):
        self.registry.append(model)

    def gridSearch(self, kf, X, y, models):

        best_models = []
        best_params = []
        classifiers = []

        for train_index, test_index in kf.split(X):

            # we only need training data for now
            X_train, _ = X.iloc[train_index], X.iloc[test_index]
            y_train, _ = y.iloc[train_index], y.iloc[test_index]

            for index in range(len(models)):
                print("Running", models[index]["name"])
                classifiers.append(models[index]["name"])
                grid_search = GridSearchCV(
                    models[index]["model"],
                    models[index]["params"],
                    cv=5,
                    scoring="f1_macro",
                    refit=True,
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(X_train, y_train)

                # extract the best estimator, params
                best_estimator = grid_search.best_estimator_
                best_param = grid_search.best_params_

                best_models.append(best_estimator)
                best_params.append(best_param)

        return best_models, best_params, X_train, y_train, classifiers
