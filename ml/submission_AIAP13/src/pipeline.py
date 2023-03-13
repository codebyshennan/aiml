import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from export import SQLite
from models import Models


class MLPipeline:
    def __init__(self, data_path, table):
        self.data = SQLite(data_path, table)
        self.models = Models()

    def preprocessing(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def training(self):
        for model in self.models.registry:
            print(model["name"])
            model.fit(self.X_train, self.y_train)

    def evaluation(self):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{name} accuracy: {accuracy}")

    def run(self):
        self.preprocessing()
        self.training()
        self.evaluation()
