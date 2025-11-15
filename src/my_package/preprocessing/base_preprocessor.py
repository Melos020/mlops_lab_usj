from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    """Abstract base class for a preprocessing step on the Titanic data."""

    @abstractmethod
    def process(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Takes raw train/test DataFrames and returns cleaned train/test."""
        raise NotImplementedError
