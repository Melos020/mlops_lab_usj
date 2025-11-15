from abc import ABC, abstractmethod
import pandas as pd


class BaseFeaturesComputer(ABC):
    """Abstract base class for feature engineering step."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a DataFrame and returns a transformed DataFrame."""
        raise NotImplementedError
