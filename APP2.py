import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Limpieza de Datos de Agricultura", layout="wide")


@dataclass
class CleaningReport:
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    num_rows_removed: int
    num_cols_casted: int
    num_missing_imputed: int
    outlier_winsorized: int
    notes: List[str]


def read_excel_dataset(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    inferred: Dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            inferred[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            inferred[col] = "numeric"
        else:
            # try coercion to numeric
            coerced = pd.to_numeric(s, errors="coerce")
            num_ratio = coerced.notna().mean()
            if num_ratio > 0.8:
                inferred[col] = "numeric"
            else:
                # try coercion to datetime
                coerced_dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                dt_ratio = coerced_dt.notna().mean()
                if dt_ratio > 0.8:
                    inferred[col] = "datetime"
                else:
                    inferred[col] = "categorical"
    return inferred


def coerce_types(df: pd.DataFrame, types: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    casted = 0
    out = df.copy()
    for col, t in types.items():
        if t == "numeric" and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            casted += 1
        elif t == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)
            casted += 1
        elif t == "categorical" and not pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype("string")
            casted += 1
    return out, casted


def winsorize_iqr(series: pd.Series, factor: float = 1.5) -> Tuple[pd.Series, int]:
    s = series.copy()
    s_clean = s.dropna()
    if s_clean.empty:
        return s, 0
    q1 = s_clean.quantile(0.25)
    q3 = s_clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    before = s.notna().sum()
    s = s.clip(lower, upper)
    after = s.notna().sum()
    # Count values that were outside bounds (approximation)
    changed = int(((series < lower) | (series > upper)).sum())
    return s, changed


def impute_missing(df: pd.DataFrame, types: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    imputed = 0
    rng = np.random.default_rng(7)

    for col, t in types.items():
        if t == "numeric":
            median_val = out[col].median(skipna=True)
            n_missing = out[col].isna().sum()
            out[col] = out[col].fillna(median_val)
            imputed += int(n_missing)
        elif t == "datetime":
            # Impute with median date (approx via numeric ordinal)
            if out[col].notna().any():
                dt = out[col].astype("datetime64[ns]")
                median_ts = pd.to_datetime(dt.dropna().astype(np.int64).median())
                n_missing = out[col].isna().sum()
                out[col] = out[col].fillna(median_ts)
                imputed += int(n_missing)
        else:  # categorical
            mode_val = out[col].mode(dropna=True)
            if not mode_val.empty:
                fill_val = mode_val.iloc[0]
            else:
                # fallback random label if all missing
                fill_val = f"cat_{rng.integers(0, 5)}"
            n_missing = out[col].isna().sum()
            out[col] = out[col].fillna(fill_val)
            imputed += int(n_missing)
    return out, imputed


def clean_agriculture_dataset(df: pd.DataFrame, iqr_factor: float = 1.5) -> Tuple[pd.DataFrame, CleaningReport]:
    notes: List[str] = []
    orig_shape = df.shape

    # Detect and coerce types
    inferred = detect_column_types(df)
    df_cast, n_casted = coerce_types(df, inferred)
    notes.append(f"Columnas convertidas de tipo: {n_casted}")

    # Winsorize numeric outliers
    total_wins = 0
    for col, t in inferred.items():
        if t == "numeric":
            df_cast[col], changed = winsorize_iqr(df_cast[col], factor=iqr_factor)
            total_wins += changed
    notes.append(f"Valores atípicos recortados (IQR): {total_wins}")

    # Impute missing values
    df_imp, n_imputed = impute_missing(df_cast, inferred)
    notes.append(f"Valores imputados: {n_imputed}")

    # Optional: drop columns completely empty after coercion
    empty_cols = [c for c in df_imp.columns if df_imp[c].isna().all()]
    if empty_cols:
        df_imp = df_imp.drop(columns=empty_cols)
        notes.append(f"Columnas eliminadas por vacías: {len(empty_cols)} -> {empty_cols}")

    final_shape = df_imp.shape
    report = CleaningReport(
        original_shape=orig_shape,
        final_shape=final_shape,
        num_rows_removed=0,
        num_cols_casted=n_casted,
        num_missing_imputed=n_imputed,
        outlier_winsorized=total_wins,
        notes=notes,
    )
    return df_imp, report


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            summary_rows.append({
                "columna": col,
                "tipo": dtype,
                "missing": missing,
                "min": desc.get("min", np.nan),
                "q1": s.quantile(0.25),
                "media": desc.get("mean", np.nan),
                "mediana": s.median(),
                "q3": s.quantile(0.75),
                "max": desc.get("max", np.nan),
            })
        else:
            n_unique = s.nunique(dropna=True)
            top = s.mode(dropna=True)
            top_val = top.iloc[0] if not top.empty else None
            summary_rows.append({
                "columna": col,
                "tipo": dtype,
                "missing": missing,
                "valores_unicos": n_unique,
                "moda": top_val,
            })
    return pd.DataFrame(summary_rows)


def render_plots(df: pd.DataFrame):
    st.subheader("Visualizaciones rápidas")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns.tolist()

    if numeric_cols:
        col = st.selectbox("Histograma de columna numérica", options=numeric_cols, key="hist_num")
        bins = st.slider("Bins", 5, 60, 20, key="bins")
        fig = px.histogram(df, x=col, nbins=bins)
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X", options=numeric_cols, key="scat_x")
        with c2:
            y = st.selectbox("Y", options=[c for c in numeric_cols if c != x], key="scat_y")
        color = st.selectbox("Color (opcional)", options=[None] + cat_cols, index=0, key="scat_color")
        fig = px.scatter(df, x=x, y=y, color=color)
        st.plotly_chart(fig, use_container_width=True)

    if cat_cols and numeric_cols:
        cat = st.selectbox("Barras por categórica", options=cat_cols, key="bar_cat")
        val = st.selectbox("Valor numérico", options=numeric_cols, key="bar_val")
        agg = st.selectbox("Agregación", ["mean", "sum", "max", "min"], index=0, key="bar_agg")
        grouped = df.groupby(cat)[val].agg(agg).reset_index()
        fig = px.bar(grouped, x=cat, y=val)
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Limpieza y EDA de Datos de Agricultura")
    st.caption("Carga el Excel, limpia valores faltantes y atípicos, y explora el dataset resultante.")

    default_path = "dataset_agricultura.xlsx"
    path = st.sidebar.text_input("Ruta del archivo Excel", value=default_path)
    iqr_factor = st.sidebar.slider("Factor IQR para recorte de outliers", 1.0, 3.0, 1.5, 0.1)
    load_btn = st.sidebar.button("Cargar y limpiar")

    if load_btn:
        try:
            raw_df = read_excel_dataset(path)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            return

        st.subheader("Vista original (primeras filas)")
        st.dataframe(raw_df.head(20), use_container_width=True)
        st.caption(f"Forma original: {raw_df.shape}")

        clean_df, report = clean_agriculture_dataset(raw_df, iqr_factor=iqr_factor)

        st.markdown("---")
        st.header("Dataset limpio")
        st.dataframe(clean_df, use_container_width=True)
        st.caption(f"Forma final: {report.final_shape}")

        st.subheader("Resumen de limpieza")
        st.write(
            f"Filas x Columnas: {report.original_shape} -> {report.final_shape} | "
            f"Casteos: {report.num_cols_casted} | Imputaciones: {report.num_missing_imputed} | "
            f"Outliers recortados: {report.outlier_winsorized}"
        )
        with st.expander("Notas"):
            for n in report.notes:
                st.write("- ", n)

        st.subheader("Resumen por columna")
        st.dataframe(dataset_summary(clean_df), use_container_width=True)

        st.markdown("---")
        render_plots(clean_df)

        csv_data = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV limpio", data=csv_data, file_name="agricultura_limpio.csv", mime="text/csv")
    else:
        st.info("Introduce la ruta del Excel y presiona 'Cargar y limpiar'. Por defecto se usará 'dataset_agricultura.xlsx'.")


if __name__ == "__main__":
    main()


