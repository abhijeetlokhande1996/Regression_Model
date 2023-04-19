from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalVarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], reference_variable: str) -> None:
        super().__init__()

        if not isinstance(variables, list):
            raise ValueError("variables should be list")

        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper"""

    def __init__(self, variables: List[str], mappings: dict) -> None:
        super().__init__()

        if not isinstance(variables, list):
            raise ValueError("variables should be list")

        self.variables = variables
        self.mapping = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mapping)
        return X
