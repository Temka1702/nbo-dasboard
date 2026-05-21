import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NBO Prototype", layout="wide")
st.title("Прототип персонализированных рекомендаций: вклад (NBO)")

DEFAULT_PATH = "artifacts/top_recommendations.csv"

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    Читаем CSV устойчиво для RU-Excel:
    - sep=';' (часто в русской локали)
    - decimal=',' чтобы 0,71 стало числом
    - encoding='utf-8-sig' чтобы корректно читались русские буквы
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",")
    # Приведение типов (на случай, если что-то прочиталось строкой)
    if "client_id" in df.columns:
        df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce")
    if "p_deposit" in df.columns:
        df["p_deposit"] = pd.to_numeric(df["p_deposit"], errors="coerce")
    return df

# -------- загрузка данных --------
st.sidebar.header("Источник данных")

use_upload = st.sidebar.checkbox("Загрузить свой CSV (вместо файла из репозитория)", value=False)

df = None
if use_upload:
    up = st.sidebar.file_uploader("Выбери CSV", type=["csv"])
    if up is not None:
        # Пытаемся читать как ; и как , в десятичных
        df = pd.read_csv(up, sep=";", encoding="utf-8-sig", decimal=",")
        if "client_id" in df.columns:
            df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce")
        if "p_deposit" in df.columns:
            df["p_deposit"] = pd.to_numeric(df["p_deposit"], errors="coerce")

if df is None:
    if not os.path.exists(DEFAULT_PATH):
        st.error(
            f"Файл не найден: {DEFAULT_PATH}\n\n"
            "Проверь, что в репозитории существует файл artifacts/top_recommendations.csv"
        )
        st.stop()
    df = load_csv(DEFAULT_PATH)

# -------- проверки колонок --------
required = {"client_id", "p_deposit"}
missing = required - set(df.columns)
if missing:
    st.error(f"Не хватает обязательных столбцов: {missing}")
    st.stop()

# -------- KPI --------
st.subheader("Ключевые показатели")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Клиентов в списке", f"{len(df):,}".replace(",", " "))
c2.metric("Средняя вероятность p", f"{df['p_deposit'].mean():.3f}")
c3.metric("Медианная вероятность p", f"{df['p_deposit'].median():.3f}")
c4.metric("Максимальная вероятность p", f"{df['p_deposit'].max():.3f}")

st.divider()

# -------- фильтры --------
st.subheader("Фильтры списка")
left, mid, right = st.columns([1, 1, 1])

top_n = left.number_input("Top-N строк", min_value=5, max_value=200000, value=100, step=5)
p_min = mid.slider("Минимальная p_deposit", 0.0, 1.0, 0.20, 0.01)

sort_mode = right.selectbox("Сортировка", ["По убыванию p", "По возрастанию p"])
ascending = (sort_mode == "По возрастанию p")

df_view = df[df["p_deposit"] >= p_min].sort_values("p_deposit", ascending=ascending).head(int(top_n))

# -------- таблица --------
st.subheader("Список клиентов для предложения вклада")
if not use_upload:
    st.caption(f"Источник: {DEFAULT_PATH}")
st.dataframe(df_view, use_container_width=True)

# -------- скачать --------
csv_out = df_view.to_csv(index=False, sep=";", encoding="utf-8-sig")
st.download_button(
    "Скачать отфильтрованный список (CSV)",
    data=csv_out.encode("utf-8-sig"),
    file_name="top_recommendations_filtered.csv",
    mime="text/csv",
)

st.divider()

# -------- график распределения --------
st.subheader("Распределение вероятности p_deposit (по всему списку)")
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(df["p_deposit"].dropna(), bins=20, edgecolor="black")
ax.set_xlabel("p_deposit")
ax.set_ylabel("Количество клиентов")
ax.grid(alpha=0.2)
st.pyplot(fig)

st.divider()

# -------- поиск клиента --------
st.subheader("Поиск клиента по ID")
min_id = int(df["client_id"].dropna().min())
max_id = int(df["client_id"].dropna().max())

cid = st.number_input("client_id", min_value=min_id, max_value=max_id, value=min_id, step=1)
row = df[df["client_id"] == cid]

if row.empty:
    st.warning("Клиент не найден в текущем списке. Возможно, он не вошёл в топ или был отфильтрован.")
else:
    st.success("Клиент найден")
    st.dataframe(row, use_container_width=True)
    st.metric("Вероятность отклика (p_deposit)", float(row["p_deposit"].iloc[0]))
