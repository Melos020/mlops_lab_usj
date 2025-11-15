from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from my_package.models.base_model import BaseModel



class TitanicLogRegModel(BaseModel):
    """Logistic Regression model for Titanic survival prediction."""

    def __init__(self, **logreg_kwargs: Any) -> None:
        # Keep model tiny: it expects already-featurized numeric data
        self.model = LogisticRegression(max_iter=1000, random_state=42, **logreg_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TitanicLogRegModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "TitanicLogRegModel":
        path = Path(path)
        instance = cls()
        instance.model = joblib.load(path)
        return instance
