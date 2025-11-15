from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from my_package.models.base_model import BaseModel


class TitanicRFModel(BaseModel):
    """Random Forest model for Titanic survival prediction."""

    def __init__(self, **rf_kwargs: Any) -> None:
        default_kwargs = dict(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        default_kwargs.update(rf_kwargs)
        self.model = RandomForestClassifier(**default_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TitanicRFModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "TitanicRFModel":
        path = Path(path)
        instance = cls()
        instance.model = joblib.load(path)
        return instance
