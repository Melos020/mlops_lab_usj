from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from typing import Any


class BaseModel(ABC):
    """Abstract base class for ML models used in this project."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model and return self."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions for the given features."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load a trained model from disk."""
        raise NotImplementedError
