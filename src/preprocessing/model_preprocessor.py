"""Transformações de ML ajustadas exclusivamente nos dados de treino."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modeling.split import get_modeling_columns


def build_model_preprocessor(
    *,
    feature_columns: list[str] | None = None,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Cria um ColumnTransformer esparso, sem aprender nada antes do fit.

    Booleanos entram como categorias porque representam estados discretos e podem
    estar ausentes. O indicador de disponibilidade permanece numérico/binário.
    """
    contract = get_modeling_columns()
    selected = set(feature_columns or contract["X"])
    numeric = [
        column for column in [*contract["numeric"], *contract["availability"]]
        if column in selected
    ]
    categorical = [
        column for column in [*contract["categorical"], *contract["boolean"]]
        if column in selected
    ]
    unknown = selected.difference(numeric, categorical)
    if unknown:
        raise ValueError(f"Features fora do contrato: {sorted(unknown)}")

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
