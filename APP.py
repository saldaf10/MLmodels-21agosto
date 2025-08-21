import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List


st.set_page_config(page_title="EDA Deportivo Sintético", layout="wide")


def get_column_pools() -> Dict[str, Dict[str, List]]:
    """Define pools of candidate columns by type for el dominio deportivo."""
    rng = np.random.default_rng(42)

    # Categóricas
    deportes = [
        "Fútbol",
        "Baloncesto",
        "Tenis",
        "Béisbol",
        "Ciclismo",
        "Atletismo",
        "Natación",
    ]
    ligas = ["Liga A", "Liga B", "Liga C", "Liga D"]
    paises = ["Colombia", "Argentina", "Brasil", "España", "EE.UU.", "México", "Chile"]
    equipos = ["Tiburones", "Leones", "Águilas", "Tigres", "Cóndores", "Guerreros"]
    posiciones = [
        "Portero",
        "Defensa",
        "Mediocampista",
        "Delantero",
        "Base",
        "Escolta",
        "Alero",
        "Pívot",
    ]
    sexo = ["Masculino", "Femenino"]
    lesionado = ["No", "Sí"]

    # Numéricas
    numeric_templates = {
        "puntos": (0, 40),
        "asistencias": (0, 15),
        "rebotes": (0, 20),
        "goles": (0, 6),
        "velocidad_kmh": (15, 40),
        "frecuencia_cardiaca": (60, 200),
        "minutos_jugados": (0, 120),
        "edad": (16, 40),
    }

    # Fechas
    date_template = {"fecha": None}

    pools = {
        "categorical": {
            "deporte": deportes,
            "liga": ligas,
            "pais": paises,
            "equipo": equipos,
            "posicion": posiciones,
            "sexo": sexo,
            "lesionado": lesionado,
        },
        "numeric": numeric_templates,
        "datetime": date_template,
    }
    return pools


def _generate_datetime_series(n_rows: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    start_date = datetime.today() - timedelta(days=365 * 2)
    dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, 365 * 2, size=n_rows)]
    dates.sort()
    return pd.to_datetime(dates)


def generate_sports_dataframe(n_rows: int, selected_cols: List[str], seed: int) -> pd.DataFrame:
    """Generate synthetic sports dataset with the requested columns."""
    rng = np.random.default_rng(seed)
    pools = get_column_pools()

    data: Dict[str, pd.Series] = {}

    for col in selected_cols:
        # Decide a pool for this column name
        if col in pools["categorical"]:
            choices = pools["categorical"][col]
            data[col] = pd.Series(rng.choice(choices, size=n_rows))
        elif col in pools["numeric"]:
            lo, hi = pools["numeric"][col]
            # Use different distributions for variety
            if col in {"puntos", "asistencias", "rebotes", "goles"}:
                vals = rng.poisson(lam=max((lo + hi) / 6, 1), size=n_rows).clip(lo, hi)
            elif col in {"velocidad_kmh", "frecuencia_cardiaca"}:
                mean = (lo + hi) / 2
                std = (hi - lo) / 6
                vals = rng.normal(mean, std, size=n_rows).clip(lo, hi)
            else:
                vals = rng.integers(lo, hi + 1, size=n_rows)
            data[col] = pd.Series(vals.astype(float))
        elif col in pools["datetime"]:
            data[col] = _generate_datetime_series(n_rows, seed)
        elif col == "indice":
            data[col] = pd.Series(np.arange(1, n_rows + 1))
        else:
            # Fallback: create a categorical with 4 random labels
            labels = [f"cat_{i}" for i in range(4)]
            data[col] = pd.Series(rng.choice(labels, size=n_rows))

    df = pd.DataFrame(data)
    return df


def limit_selection(selection: List[str], max_cols: int) -> List[str]:
    if len(selection) > max_cols:
        st.warning(f"Seleccionaste {len(selection)} columnas. Se limitarán a {max_cols} primeras.")
        return selection[:max_cols]
    return selection


def sidebar_controls():
    st.sidebar.title("Configuración de datos")

    n_rows = st.sidebar.slider("Número de muestras", min_value=50, max_value=500, value=200, step=10)
    max_cols = st.sidebar.slider("Número de columnas (máx 6)", min_value=1, max_value=6, value=4)
    seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=10_000_000, value=42, step=1)

    tipo = st.sidebar.selectbox("Tipo de variables", ["Cuantitativas", "Cualitativas", "Mixtas"], index=2)

    pools = get_column_pools()
    quant_cols = list(pools["numeric"].keys()) + ["indice"]
    cat_cols = list(pools["categorical"].keys())
    date_cols = list(pools["datetime"].keys())

    if tipo == "Cuantitativas":
        candidates = quant_cols + date_cols
    elif tipo == "Cualitativas":
        candidates = cat_cols
    else:
        candidates = quant_cols + cat_cols + date_cols

    default_selection = [c for c in ["fecha", "deporte", "equipo", "puntos"] if c in candidates][:max_cols]
    selected_cols = st.sidebar.multiselect(
        "Selecciona columnas (máx 6)", options=candidates, default=default_selection,
        help="Elige las columnas a incluir en el dataset sintetizado"
    )
    selected_cols = limit_selection(selected_cols, max_cols)

    regenerate = st.sidebar.button("Generar datos")

    return n_rows, selected_cols, seed, regenerate


def render_dataset(df: pd.DataFrame):
    st.subheader("Vista de datos")
    st.dataframe(df, use_container_width=True)

    st.caption(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

    numeric_df = df.select_dtypes(include=["number"])  # floats/ints
    if not numeric_df.empty:
        st.subheader("Estadísticos descriptivos")
        st.dataframe(numeric_df.describe().T, use_container_width=True)

        st.subheader("Matriz de correlación")
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name="datos_deportivos_sinteticos.csv", mime="text/csv")


def render_charts(df: pd.DataFrame):
    st.header("Visualizaciones")
    chart_types = st.multiselect(
        "Elige las gráficas a mostrar",
        ["Línea", "Barras", "Dispersión", "Pastel", "Histograma"],
        default=["Barras", "Histograma"],
    )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # Línea
    if "Línea" in chart_types:
        st.subheader("Gráfica de línea")
        x_options = date_cols + numeric_cols + [c for c in df.columns if c not in numeric_cols and c not in date_cols]
        x_col = st.selectbox("Eje X", options=x_options, index=0 if date_cols else 0, key="line_x")
        y_col = st.selectbox("Eje Y (numérico)", options=numeric_cols, key="line_y")
        color_col = st.selectbox("Color (opcional)", options=[None] + categorical_cols, index=0, key="line_color")
        if y_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Se requiere al menos una columna numérica para el eje Y.")

    # Barras
    if "Barras" in chart_types:
        st.subheader("Gráfica de barras")
        bar_mode = st.radio("Modo", ["Conteos", "Agregado"], horizontal=True, key="bar_mode")
        if bar_mode == "Conteos":
            if categorical_cols:
                x_cat = st.selectbox("Columna categórica", options=categorical_cols, key="bar_count_x")
                fig = px.bar(df, x=x_cat)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columnas categóricas para mostrar conteos.")
        else:
            if categorical_cols and numeric_cols:
                x_cat = st.selectbox("Agrupar por (categórica)", options=categorical_cols, key="bar_agg_x")
                y_num = st.selectbox("Valor (numérico)", options=numeric_cols, key="bar_agg_y")
                agg_fn = st.selectbox("Agregación", ["media", "suma", "máximo", "mínimo"], key="bar_agg_fn")
                agg_map = {"media": "mean", "suma": "sum", "máximo": "max", "mínimo": "min"}
                grouped = df.groupby(x_cat)[y_num].agg(agg_map[agg_fn]).reset_index()
                fig = px.bar(grouped, x=x_cat, y=y_num)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Se requieren columnas categóricas y numéricas para agregación.")

    # Dispersión
    if "Dispersión" in chart_types:
        st.subheader("Gráfico de dispersión")
        if len(numeric_cols) >= 2:
            x_num = st.selectbox("Eje X (numérico)", options=numeric_cols, key="scat_x")
            y_num = st.selectbox("Eje Y (numérico)", options=[c for c in numeric_cols if c != x_num], key="scat_y")
            color_col = st.selectbox("Color (opcional)", options=[None] + categorical_cols, index=0, key="scat_color")
            fig = px.scatter(df, x=x_num, y=y_num, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Se requieren al menos dos columnas numéricas para la dispersión.")

    # Pastel
    if "Pastel" in chart_types:
        st.subheader("Gráfico de pastel")
        if categorical_cols:
            names_col = st.selectbox("Categórica", options=categorical_cols, key="pie_names")
            pie_mode = st.radio("Valores", ["Conteos", "Numérica agregada"], horizontal=True, key="pie_mode")
            if pie_mode == "Conteos":
                fig = px.pie(df, names=names_col)
            else:
                if numeric_cols:
                    values_col = st.selectbox("Valor (numérico)", options=numeric_cols, key="pie_values")
                    grouped = df.groupby(names_col)[values_col].sum().reset_index()
                    fig = px.pie(grouped, names=names_col, values=values_col)
                else:
                    st.info("No hay columnas numéricas para agregar valores; se usarán conteos.")
                    fig = px.pie(df, names=names_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay columnas categóricas para graficar.")

    # Histograma
    if "Histograma" in chart_types:
        st.subheader("Histograma")
        if numeric_cols:
            x_num = st.selectbox("Columna (numérica)", options=numeric_cols, key="hist_x")
            bins = st.slider("Bins", min_value=5, max_value=60, value=20)
            color_col = st.selectbox("Color (opcional)", options=[None] + categorical_cols, index=0, key="hist_color")
            fig = px.histogram(df, x=x_num, nbins=bins, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay columnas numéricas para histograma.")


def main():
    st.title("EDA con Datos Sintéticos de Deportes")
    st.write(
        "Genera un conjunto de datos sintético con variables deportivas y explóralo con diferentes visualizaciones. "
        "Puedes elegir hasta 500 muestras y hasta 6 columnas, incluyendo variables cuantitativas y cualitativas."
    )

    n_rows, selected_cols, seed, regenerate = sidebar_controls()

    if regenerate or "df" not in st.session_state:
        st.session_state["df"] = generate_sports_dataframe(n_rows, selected_cols, seed)
    else:
        # Si cambian controles sin pulsar el botón, mantenemos el df actual
        pass

    df = st.session_state["df"]

    st.markdown("---")
    render_dataset(df)
    st.markdown("---")
    render_charts(df)


if __name__ == "__main__":
    main()


