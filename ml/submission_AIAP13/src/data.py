from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# data helpers
class Data:
    def __init__(self, n_splits=5):
        self.random_state = 123
        self.n_splits = n_splits
        self.numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", MinMaxScaler()),
            ]
        )
        self.categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

    def trainTestSplit(self, df, features, targets):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        X, y = df[features], df[targets]
        return kf, X, y


# full_processor = ColumnTransformer(transformers=[
#     ('number', numeric_pipeline, numerical_features),
#     ('category', categorical_pipeline, categorical_features)
# ])
