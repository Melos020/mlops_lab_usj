import pandas as pd

from .base_preprocessor import BasePreprocessor


class TitanicPreprocessor(BasePreprocessor):
    """Concrete preprocessor for the Titanic Kaggle dataset."""

    def process(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Clean train and test DataFrames and return the processed versions."""

        train = train.copy()
        test = test.copy()

        for df in (train, test):
            if "Cabin" in df.columns:
                df.drop(columns=["Cabin"], inplace=True)

        if "Embarked" in train.columns:
            train["Embarked"].fillna("S", inplace=True)
        if "Fare" in test.columns:
            test["Fare"].fillna(test["Fare"].mean(), inplace=True)

        n_train = len(train)
        df_all = pd.concat([train, test], sort=True).reset_index(drop=True)

        if {"Sex", "Pclass", "Age"}.issubset(df_all.columns):
            df_all["Age"] = df_all.groupby(["Sex", "Pclass"])["Age"].transform(
                lambda x: x.fillna(x.median())
            )
            
        train_proc = df_all.iloc[:n_train].copy()
        test_proc = df_all.iloc[n_train:].copy()

        if "Survived" in test_proc.columns:
            test_proc.drop(columns=["Survived"], inplace=True)
        if "Survived" in train_proc.columns:
            train_proc["Survived"] = train_proc["Survived"].astype("int64")

        return train_proc, test_proc
