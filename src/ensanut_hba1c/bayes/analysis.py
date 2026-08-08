"""Interpretable one-versus-rest Bayes score engine.

The module converts numeric variables into robust adaptive bins and categorical
variables into indicator columns. Each observed bin/category receives a
smoothed log-likelihood-ratio score comparing the target class with the rest.
The row-level score is the sum of all active bin/category scores.

The analysis is descriptive when fitted and evaluated on the same observations.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

BIN_SEP = "__BAYES_BIN__"


def parsear_values(value: Any) -> dict[str, Any]:
    """Parse the ENSANUT dictionary ``values`` field into a dictionary."""

    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def normalizar_codigo(value: Any) -> str | None:
    """Normalize coded numeric/string values without changing their meaning."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        numeric = float(text)
        if np.isfinite(numeric):
            if numeric.is_integer():
                return str(int(numeric))
            return format(numeric, ".15g")
    except ValueError:
        pass
    return text


def normalizar_selector_bin(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "")
    if text.startswith("bin"):
        suffix = text[3:].split("/")[0].split("(")[0]
        if suffix.isdigit():
            return f"bin{int(suffix)}"
    return text


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 1000 or (0 < abs(float(value)) < 0.001):
        return f"{float(value):.4g}"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _interval_text(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "NO_DATA"
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        return _format_number(minimum)
    return f"[{_format_number(minimum)}, {_format_number(maximum)}]"


def _labels_from_codes(codes: pd.Series, prefix_start: int = 1) -> tuple[pd.Series, dict[str, str], list[str]]:
    unique_codes = sorted(codes.dropna().unique())
    code_to_bin = {
        code: f"bin{index}"
        for index, code in enumerate(unique_codes, start=prefix_start)
    }
    labels = codes.map(code_to_bin).astype("string")
    bins = list(code_to_bin.values())
    ranges = {
        bin_name: _interval_text(codes.loc[codes.eq(code)])
        for code, bin_name in code_to_bin.items()
    }
    return labels, ranges, bins


def construir_bins_fijos(
    serie: pd.Series,
    n_bins: int = 10,
    force_exact: bool = False,
) -> tuple[pd.Series, dict[str, str], list[str]]:
    """Create adaptive numeric bins without artificial empty bins.

    Rules
    -----
    1. Variables with at most ``n_bins`` unique values use one bin per value.
    2. Non-negative zero-inflated variables reserve ``bin1`` for zero and use
       quantiles of positive values for the remaining bins.
    3. Other continuous variables use quantiles while retaining tied values.
    4. ``force_exact=True`` ranks observations first, which can split ties. It
       is available for diagnostics but is not recommended for interpretation.
    """

    n_bins = max(int(n_bins), 1)
    numeric = pd.to_numeric(serie, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(pd.NA, index=serie.index, dtype="string")
    valid = numeric.dropna()
    if valid.empty:
        return result, {}, []

    unique_count = int(valid.nunique())
    if unique_count <= n_bins and not force_exact:
        labels, ranges, bins = _labels_from_codes(numeric)
        return labels, ranges, bins

    def quantile_labels(values: pd.Series, q: int, start: int = 1) -> tuple[pd.Series, dict[str, str], list[str]]:
        q = max(1, min(int(q), int(values.nunique()), len(values)))
        if q == 1:
            labels_local = pd.Series(f"bin{start}", index=values.index, dtype="string")
        else:
            source = values.rank(method="first") if force_exact else values
            try:
                cut = pd.qcut(source, q=q, labels=False, duplicates="drop")
            except ValueError:
                cut = pd.Series(0, index=values.index)
            labels_local = cut.map(
                lambda code: f"bin{int(code) + start}" if pd.notna(code) else pd.NA
            ).astype("string")
        bins_local = sorted(
            labels_local.dropna().unique(),
            key=lambda label: int(str(label).replace("bin", "")),
        )
        ranges_local = {
            label: _interval_text(values.loc[labels_local.eq(label)])
            for label in bins_local
        }
        return labels_local, ranges_local, bins_local

    nonnegative = bool(valid.min() >= 0)
    zero_count = int(valid.eq(0).sum())
    positive = valid.loc[valid.gt(0)]
    zero_inflated = (
        nonnegative
        and zero_count > 0
        and not positive.empty
        and zero_count / len(valid) >= 0.05
        and n_bins >= 2
    )

    if zero_inflated:
        result.loc[numeric.eq(0)] = "bin1"
        ranges = {"bin1": "0"}
        positive_labels, positive_ranges, positive_bins = quantile_labels(
            positive, q=n_bins - 1, start=2
        )
        result.loc[positive_labels.index] = positive_labels
        ranges.update(positive_ranges)
        bins = ["bin1", *positive_bins]
        return result, ranges, bins

    labels, ranges, bins = quantile_labels(valid, q=n_bins, start=1)
    result.loc[labels.index] = labels
    return result, ranges, bins


@dataclass
class BayesClassifier:
    """Smoothed additive log-likelihood-ratio classifier."""

    alpha: float = 1.0
    min_cases: int = 5
    sep: str = BIN_SEP

    def fit(self, X_bin: pd.DataFrame, y: Sequence[int]):
        """Fit the additive Bayes score using vectorized contingency counts."""

        if len(X_bin) != len(y):
            raise ValueError("X_bin and y must contain the same number of rows.")
        y_array = np.asarray(y, dtype=np.uint8)
        if set(np.unique(y_array)).difference({0, 1}):
            raise ValueError("BayesClassifier requires a binary target coded 0/1.")
        if len(np.unique(y_array)) < 2:
            raise ValueError("Both target classes are required.")

        X = X_bin.fillna(0).astype(np.uint8)
        X_array = X.to_numpy(dtype=np.uint8, copy=False)
        positive_mask = y_array == 1
        negative_mask = ~positive_mask

        N = int(len(y_array))
        Nc = int(positive_mask.sum())
        Nnc = int(negative_mask.sum())
        prior = Nc / N

        # All binary-feature counts are computed in vectorized form. This avoids
        # a Python loop over thousands of dummy/bin columns for every cluster.
        Nx = X_array.sum(axis=0, dtype=np.int64)
        nCx = X_array[positive_mask].sum(axis=0, dtype=np.int64)
        nnCx = X_array[negative_mask].sum(axis=0, dtype=np.int64)

        p_x_given_c = (nCx + self.alpha) / (Nc + 2 * self.alpha)
        p_x_given_nc = (nnCx + self.alpha) / (Nnc + 2 * self.alpha)
        likelihood_ratio = p_x_given_c / p_x_given_nc
        raw_score = np.log(likelihood_ratio)
        score_aplicado = (Nx >= self.min_cases).astype(np.uint8)
        score = np.where(score_aplicado == 1, raw_score, 0.0)
        p_c_given_x = (nCx + self.alpha) / (Nx + 2 * self.alpha)

        variables: list[str] = []
        raw_values: list[str] = []
        for column in X.columns.astype(str):
            if self.sep in column:
                variable, raw_value = column.split(self.sep, 1)
            else:
                variable, raw_value = column, "1"
            variables.append(variable)
            raw_values.append(raw_value)

        operations = pd.DataFrame(
            {
                "var": variables,
                "valor_crudo": raw_values,
                "N": N,
                "Nc": Nc,
                "Nnc": Nnc,
                "prior": prior,
                "alpha": self.alpha,
                "min_cases": self.min_cases,
                "Nx": Nx.astype(np.int64),
                "nCx": nCx.astype(np.int64),
                "nnCx": nnCx.astype(np.int64),
                "p_x_given_c": p_x_given_c,
                "p_x_given_nc": p_x_given_nc,
                "p_c_given_x": p_c_given_x,
                "likelihood_ratio": likelihood_ratio,
                "raw_log_likelihood_score": raw_score,
                "score_aplicado": score_aplicado,
                "score": score,
                "binary_column": X.columns.astype(str),
            }
        )

        self.operations_table_ = operations
        self.feature_table_ = operations[
            [
                "var",
                "valor_crudo",
                "Nx",
                "nCx",
                "nnCx",
                "p_c_given_x",
                "likelihood_ratio",
                "score_aplicado",
                "score",
                "binary_column",
            ]
        ].copy()
        self.score_by_column_ = operations.set_index("binary_column")["score"]
        self.columns_ = X.columns.tolist()
        self.prior_ = prior
        return self

    def decision_function(self, X_bin: pd.DataFrame) -> np.ndarray:
        X = X_bin.reindex(columns=self.columns_, fill_value=0).fillna(0).astype(float)
        scores = self.score_by_column_.reindex(self.columns_).fillna(0).to_numpy()
        return X.to_numpy() @ scores

    def predict_proba(self, X_bin: pd.DataFrame) -> np.ndarray:
        score = self.decision_function(X_bin)
        prior_logit = math.log(self.prior_ / (1 - self.prior_))
        posterior = 1.0 / (1.0 + np.exp(-(prior_logit + score)))
        return np.column_stack([1 - posterior, posterior])


def _dictionary_row_map(df_dict: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["var"]): row.to_dict()
        for _, row in df_dict.drop_duplicates("var", keep="first").iterrows()
        if pd.notna(row.get("var"))
    }


def _infer_dictionary(df_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df_data.columns:
        series = df_data[column]
        is_category = not pd.api.types.is_numeric_dtype(series)
        values: dict[str, Any] = {"is_category": "true" if is_category else "false"}
        if is_category and series.nunique(dropna=True) <= 100:
            values["options"] = {
                normalizar_codigo(value): str(value)
                for value in series.dropna().unique()
                if normalizar_codigo(value) is not None
            }
        rows.append(
            {
                "subcategoria": "INFERRED",
                "subseccion": "INFERRED",
                "var": column,
                "var_alias": column,
                "description": column,
                "values": values,
            }
        )
    return pd.DataFrame(rows)


def construir_target_binario(
    df_data: pd.DataFrame,
    df_dict: pd.DataFrame,
    target_var: str,
    target_selector: Any,
    n_bins_target: int = 10,
    force_exact_target_bins: bool = False,
):
    """Construct a binary target from categorical choices or numeric bins."""

    if target_var not in df_data.columns:
        raise KeyError(f"Target variable not found: {target_var}")
    selection = (
        list(target_selector)
        if isinstance(target_selector, (list, tuple, set, np.ndarray, pd.Series))
        else [target_selector]
    )
    row = df_dict.loc[df_dict["var"].astype(str).eq(str(target_var))]
    values_info = parsear_values(row.iloc[0].get("values", {})) if not row.empty else {}
    is_category = str(values_info.get("is_category", "false")).lower() == "true"

    if is_category:
        options = values_info.get("options", {}) or {}
        options_norm = {
            normalizar_codigo(code): str(label)
            for code, label in options.items()
            if normalizar_codigo(code) is not None
        }
        selected_codes: set[str] = set()
        selected_texts: set[str] = set()
        for selected in selection:
            selected_code = normalizar_codigo(selected)
            if selected_code is not None:
                selected_codes.add(selected_code)
            selected_text = str(selected).strip()
            selected_texts.add(selected_text.lower())
            for code, label in options_norm.items():
                if label.strip().lower() == selected_text.lower():
                    selected_codes.add(code)

        raw = df_data[target_var]
        normalized = raw.map(normalizar_codigo)
        mask = normalized.isin(selected_codes) | raw.astype(str).str.strip().str.lower().isin(
            selected_texts
        )
        target = pd.Series(
            np.where(raw.isna(), np.nan, mask.astype(int)),
            index=df_data.index,
            name="__target_bin__",
        )
        description = {
            "target_var": target_var,
            "tipo": "categorica",
            "seleccion": selection,
            "codes_detectados": sorted(selected_codes),
        }
        return target, description

    numeric = pd.to_numeric(df_data[target_var], errors="coerce")
    bin_series, ranges, bins_with_data = construir_bins_fijos(
        numeric, n_bins=n_bins_target, force_exact=force_exact_target_bins
    )
    selected_bins = [normalizar_selector_bin(selected) for selected in selection]
    mask = bin_series.isin(selected_bins)
    target = pd.Series(
        np.where(numeric.isna(), np.nan, mask.astype(int)),
        index=df_data.index,
        name="__target_bin__",
    )
    description = {
        "target_var": target_var,
        "tipo": "continua",
        "seleccion": selected_bins,
        "n_bins_solicitados": n_bins_target,
        "bins_con_datos": bins_with_data,
        "rangos": ranges,
        "force_exact_target_bins": force_exact_target_bins,
    }
    return target, description


def preparar_diccionario_bayes(
    df_data: pd.DataFrame,
    df_dict: pd.DataFrame | None,
) -> pd.DataFrame:
    """Normalize the variable dictionary used by the Bayes encoder."""

    dictionary = _infer_dictionary(df_data) if df_dict is None else df_dict.copy()
    dictionary = dictionary.loc[:, ~dictionary.columns.duplicated()].copy()
    if "var" not in dictionary.columns:
        raise KeyError("The dictionary requires a 'var' column.")
    for required in ["subcategoria", "subseccion", "var_alias", "description", "values"]:
        if required not in dictionary.columns:
            dictionary[required] = (
                "" if required != "values" else [{} for _ in range(len(dictionary))]
            )
    return (
        dictionary
        .drop_duplicates(subset=["var"], keep="first")
        .reset_index(drop=True)
    )


def codificar_predictores_bayes(
    df_data: pd.DataFrame,
    df_dict: pd.DataFrame | None,
    *,
    excluded_variables: Iterable[str] | None = None,
    n_bins_features: int = 10,
    force_exact_feature_bins: bool = False,
    max_categorias_inferidas: int = 20,
) -> dict[str, Any]:
    """Encode all predictors exactly once for one analytical variable group.

    The returned binary matrix can be reused for every one-versus-rest cluster
    target. Numeric bins, categorical dummy columns, metadata labels, and
    diagnostics therefore remain identical across all clusters in the group.
    """

    dictionary = preparar_diccionario_bayes(df_data, df_dict)
    excluded = {str(value) for value in (excluded_variables or [])}
    working = df_data.copy()

    binary_parts: list[pd.DataFrame] = []
    label_map: dict[tuple[str, str], str] = {}
    type_map: dict[str, str] = {}
    encoding_diagnostics: list[dict[str, Any]] = []

    for _, row in dictionary.iterrows():
        variable = str(row["var"])
        if variable not in working.columns or variable in excluded:
            continue

        values_info = parsear_values(row.get("values", {}))
        declared_category = (
            str(values_info.get("is_category", "false")).strip().lower() == "true"
        )
        normalized = working[variable].map(normalizar_codigo)
        unique_count = int(normalized.nunique(dropna=True))
        non_missing_count = int(normalized.notna().sum())
        numeric = pd.to_numeric(working[variable], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        numeric_count = int(numeric.notna().sum())

        diagnostic = {
            "var": variable,
            "declared_category": declared_category,
            "non_missing_count": non_missing_count,
            "unique_count": unique_count,
            "numeric_count": numeric_count,
            "encoding": None,
            "status": "skipped",
            "reason": "",
        }

        use_categorical = declared_category or (
            numeric_count == 0
            and 1 <= unique_count <= max_categorias_inferidas
        )

        if use_categorical:
            if unique_count == 0:
                diagnostic["reason"] = "all values are missing"
                encoding_diagnostics.append(diagnostic)
                continue
            if unique_count > max_categorias_inferidas:
                diagnostic["reason"] = (
                    f"{unique_count} categories exceed max_categorias_inferidas="
                    f"{max_categorias_inferidas}"
                )
                encoding_diagnostics.append(diagnostic)
                continue

            dummies = pd.get_dummies(
                normalized,
                prefix=variable,
                prefix_sep=BIN_SEP,
                dtype=np.uint8,
            )
            if dummies.empty:
                diagnostic["reason"] = "dummy encoding produced no columns"
                encoding_diagnostics.append(diagnostic)
                continue

            options = values_info.get("options", {}) or {}
            options_norm = {
                normalizar_codigo(code): str(label)
                for code, label in options.items()
                if normalizar_codigo(code) is not None
            }
            for code in normalized.dropna().unique():
                code_string = str(code)
                label = options_norm.get(code_string, code_string)
                label_map[(variable, code_string)] = f"{code_string}: {label}"

            inferred = not declared_category
            type_map[variable] = (
                "categorical_inferred" if inferred else "categorical"
            )
            diagnostic.update(
                {
                    "encoding": type_map[variable],
                    "status": "encoded",
                    "reason": "",
                    "n_binary_columns": int(dummies.shape[1]),
                }
            )
            encoding_diagnostics.append(diagnostic)
            binary_parts.append(dummies)
            continue

        if numeric_count == 0:
            if unique_count > max_categorias_inferidas:
                diagnostic["reason"] = (
                    "non-numeric variable has too many categories to infer "
                    f"({unique_count} > {max_categorias_inferidas})"
                )
            else:
                diagnostic["reason"] = "no numeric or categorical values available"
            encoding_diagnostics.append(diagnostic)
            continue

        binned, ranges, bins_with_data = construir_bins_fijos(
            numeric,
            n_bins=n_bins_features,
            force_exact=force_exact_feature_bins,
        )
        if not bins_with_data:
            diagnostic["reason"] = "numeric binning produced no populated bins"
            encoding_diagnostics.append(diagnostic)
            continue

        dummies = pd.get_dummies(
            binned,
            prefix=variable,
            prefix_sep=BIN_SEP,
            dtype=np.uint8,
        )
        expected_columns = [
            f"{variable}{BIN_SEP}{bin_name}" for bin_name in bins_with_data
        ]
        dummies = dummies.reindex(columns=expected_columns, fill_value=0)
        for bin_name in bins_with_data:
            label_map[(variable, bin_name)] = (
                f"{bin_name}/{len(bins_with_data)} "
                f"({ranges.get(bin_name, 'NO_DATA')})"
            )
        type_map[variable] = "continuous"
        diagnostic.update(
            {
                "encoding": "continuous",
                "status": "encoded",
                "reason": "",
                "n_binary_columns": int(dummies.shape[1]),
            }
        )
        encoding_diagnostics.append(diagnostic)
        binary_parts.append(dummies)

    if not binary_parts:
        diagnostics_frame = pd.DataFrame(encoding_diagnostics)
        reason_counts = (
            diagnostics_frame["reason"].value_counts().to_dict()
            if not diagnostics_frame.empty and "reason" in diagnostics_frame
            else {}
        )
        raise ValueError(
            "No binary predictors could be generated for the Bayes analysis. "
            f"Variables inspected: {len(encoding_diagnostics)}. "
            f"Skip reasons: {reason_counts}"
        )

    X_bin = pd.concat(binary_parts, axis=1).fillna(0).astype(np.uint8)
    X_bin = X_bin.loc[:, ~X_bin.columns.duplicated()].copy()

    diagnostics_frame = pd.DataFrame(encoding_diagnostics)
    return {
        "X_bin": X_bin,
        "dictionary": dictionary,
        "label_map": label_map,
        "type_map": type_map,
        "encoding_diagnostics": diagnostics_frame,
        "row_index": working.index.to_numpy(),
        "binary_matrix_shape": X_bin.shape,
        "encoding_passes": 1,
        "encoded_variables": int(
            diagnostics_frame["status"].eq("encoded").sum()
        ) if not diagnostics_frame.empty else 0,
    }


def evaluar_target_desde_predictores_codificados(
    *,
    encoded_predictors: Mapping[str, Any],
    y: Sequence[int],
    target_info: Mapping[str, Any],
    alpha: float = 1.0,
    min_cases: int = 5,
) -> dict[str, Any]:
    """Fit and score one binary target using a previously encoded matrix."""

    X_bin = encoded_predictors["X_bin"]
    y_array = np.asarray(y, dtype=np.uint8)
    if len(X_bin) != len(y_array):
        raise ValueError("Encoded predictors and target must have the same rows.")
    if len(np.unique(y_array)) < 2:
        raise ValueError("The binary target must contain both classes.")

    dictionary = encoded_predictors["dictionary"]
    dictionary_map = _dictionary_row_map(dictionary)
    label_map = encoded_predictors["label_map"]
    type_map = encoded_predictors["type_map"]

    model = BayesClassifier(alpha=alpha, min_cases=min_cases, sep=BIN_SEP).fit(
        X_bin, y_array
    )
    operations = model.operations_table_.copy()

    metadata_rows = []
    for _, operation in operations.iterrows():
        variable = str(operation["var"])
        raw_value = str(operation["valor_crudo"])
        metadata = dictionary_map.get(variable, {})
        metadata_rows.append(
            {
                "subcategoria": metadata.get("subcategoria", ""),
                "subseccion": metadata.get("subseccion", ""),
                "var": variable,
                "var_alias": metadata.get("var_alias", variable),
                "description": metadata.get("description", variable),
                "tipo_variable": type_map.get(variable, "unknown"),
                "values_diccionario": metadata.get("values", {}),
                "categoria_o_rango": label_map.get((variable, raw_value), raw_value),
                "valor_crudo": raw_value,
            }
        )
    metadata_frame = pd.DataFrame(metadata_rows)
    operations = pd.concat(
        [
            metadata_frame.reset_index(drop=True),
            operations.drop(columns=["var", "valor_crudo"]).reset_index(drop=True),
        ],
        axis=1,
    )
    report_columns = [
        "subcategoria",
        "subseccion",
        "var",
        "var_alias",
        "description",
        "tipo_variable",
        "values_diccionario",
        "categoria_o_rango",
        "valor_crudo",
        "Nx",
        "nCx",
        "nnCx",
        "p_c_given_x",
        "likelihood_ratio",
        "score_aplicado",
        "score",
    ]
    report = operations[report_columns].copy()
    report = report.sort_values(
        ["score_aplicado", "score"], ascending=[False, False]
    ).reset_index(drop=True)

    row_scores = model.decision_function(X_bin)
    row_probabilities = model.predict_proba(X_bin)[:, 1]
    row_score_table = pd.DataFrame(
        {
            "row_index": np.asarray(encoded_predictors["row_index"]),
            "target": y_array,
            "bayes_score": row_scores,
            "bayes_probability": row_probabilities,
        }
    )

    return {
        "scores": report,
        "operaciones": operations,
        "target_info": dict(target_info),
        "row_scores": row_score_table,
        "binary_matrix_shape": X_bin.shape,
        "model": model,
        "encoding_diagnostics": encoded_predictors["encoding_diagnostics"].copy(),
        "encoding_passes": int(encoded_predictors.get("encoding_passes", 1)),
    }


def ejecutar_analisis_completo(
    df_data: pd.DataFrame,
    df_dict: pd.DataFrame | None,
    target_var: str,
    target_selector: Any,
    n_bins_target: int = 10,
    n_bins_features: int = 10,
    alpha: float = 1.0,
    min_cases: int = 5,
    force_exact_target_bins: bool = False,
    force_exact_feature_bins: bool = False,
    max_categorias_inferidas: int = 20,
):
    """Run one binary-target analysis using the reusable encoding pipeline."""

    dictionary = preparar_diccionario_bayes(df_data, df_dict)
    target_bin, target_info = construir_target_binario(
        df_data=df_data,
        df_dict=dictionary,
        target_var=target_var,
        target_selector=target_selector,
        n_bins_target=n_bins_target,
        force_exact_target_bins=force_exact_target_bins,
    )

    valid_mask = target_bin.notna()
    working = df_data.loc[valid_mask].copy()
    y = target_bin.loc[valid_mask].astype(np.uint8).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"Target {target_var} does not contain both classes after selection.")

    encoded = codificar_predictores_bayes(
        working,
        dictionary,
        excluded_variables={target_var, "__target_bin__"},
        n_bins_features=n_bins_features,
        force_exact_feature_bins=force_exact_feature_bins,
        max_categorias_inferidas=max_categorias_inferidas,
    )
    return evaluar_target_desde_predictores_codificados(
        encoded_predictors=encoded,
        y=y,
        target_info=target_info,
        alpha=alpha,
        min_cases=min_cases,
    )


__all__ = [
    "BIN_SEP",
    "parsear_values",
    "normalizar_codigo",
    "construir_bins_fijos",
    "construir_target_binario",
    "preparar_diccionario_bayes",
    "codificar_predictores_bayes",
    "evaluar_target_desde_predictores_codificados",
    "BayesClassifier",
    "ejecutar_analisis_completo",
]
