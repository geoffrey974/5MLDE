"""Prétraitement : sélection des features et encodage, intégrés au pipeline du modèle."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def split_features_target(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les features, listées dans la config, de la cible."""
    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    return df[feature_cols], df[cfg["target"]]

def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Encode les variables catégorielles en one-hot et laisse passer les numériques."""
    return ColumnTransformer(
        transformers=[
            # handle_unknown="ignore" : une catégorie jamais vue à l'entraînement ne fait pas planter la prédiction
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["features"]["categorical"]),
            ("num", "passthrough", cfg["features"]["numeric"]),
        ]
    )