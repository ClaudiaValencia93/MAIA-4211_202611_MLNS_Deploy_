"""
streamlit_app.py
Aplicación Streamlit para clasificación de textos según los ODS de la ONU
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px

# Agregar src al path para importar módulos locales
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ModelController import ModelController, ODS_NAMES

# Colores e íconos oficiales de cada ODS
ODS_INFO = {
    1:  {"color": "#E5243B", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Sustainable_Development_Goal_1.png/200px-Sustainable_Development_Goal_1.png"},
    2:  {"color": "#DDA63A", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Sustainable_Development_Goal_2.png/200px-Sustainable_Development_Goal_2.png"},
    3:  {"color": "#4C9F38", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Sustainable_Development_Goal_3.png/200px-Sustainable_Development_Goal_3.png"},
    4:  {"color": "#C5192D", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Sustainable_Development_Goal_4.png/200px-Sustainable_Development_Goal_4.png"},
    5:  {"color": "#FF3A21", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sustainable_Development_Goal_5.png/200px-Sustainable_Development_Goal_5.png"},
    6:  {"color": "#26BDE2", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Sustainable_Development_Goal_6.png/200px-Sustainable_Development_Goal_6.png"},
    7:  {"color": "#FCC30B", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Sustainable_Development_Goal_7.png/200px-Sustainable_Development_Goal_7.png"},
    8:  {"color": "#A21942", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Sustainable_Development_Goal_8.png/200px-Sustainable_Development_Goal_8.png"},
    9:  {"color": "#FD6925", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Sustainable_Development_Goal_9.png/200px-Sustainable_Development_Goal_9.png"},
    10: {"color": "#DD1367", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Sustainable_Development_Goal_10.png/200px-Sustainable_Development_Goal_10.png"},
    11: {"color": "#FD9D24", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Sustainable_Development_Goal_11.png/200px-Sustainable_Development_Goal_11.png"},
    12: {"color": "#BF8B2E", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Sustainable_Development_Goal_12.png/200px-Sustainable_Development_Goal_12.png"},
    13: {"color": "#3F7E44", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Sustainable_Development_Goal_13.png/200px-Sustainable_Development_Goal_13.png"},
    14: {"color": "#0A97D9", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Sustainable_Development_Goal_14.png/200px-Sustainable_Development_Goal_14.png"},
    15: {"color": "#56C02B", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Sustainable_Development_Goal_15.png/200px-Sustainable_Development_Goal_15.png"},
    16: {"color": "#00689D", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Sustainable_Development_Goal_16.png/200px-Sustainable_Development_Goal_16.png"},
    17: {"color": "#19486A", "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Sustainable_Development_Goal_17.png/200px-Sustainable_Development_Goal_17.png"},
}

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de ODS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Carga del modelo (cacheado para eficiencia)
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resources", "model.joblib")

@st.cache_resource
def load_model():
    return ModelController(MODEL_PATH)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Sustainable_Development_Goals.jpg/640px-Sustainable_Development_Goals.jpg",
             use_container_width=True)
    st.markdown("## 🌍 Clasificador de ODS")
    st.markdown(
        "Esta aplicación clasifica textos según los **17 Objetivos de Desarrollo Sostenible (ODS)** "
        "de la Agenda 2030 de la ONU, usando un modelo de Machine Learning entrenado con "
        "Regresión Logística y TF-IDF."
    )
    st.markdown("---")
    st.markdown("**Modelo:** Regresión Logística + TF-IDF + TruncatedSVD")
    st.markdown("**F1-Score (test):** 85.83%")
    st.markdown("---")
    st.markdown("**Integrantes del equipo:**")
    st.markdown("*(Agrega aquí los nombres)*")

# ─────────────────────────────────────────────
# Título principal
# ─────────────────────────────────────────────
st.title("Clasificador de Textos por ODS")
st.markdown(
    "Ingresa un texto relacionado con los Objetivos de Desarrollo Sostenible y el modelo "
    "predecirá a cuál ODS pertenece."
)

# ─────────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────────
try:
    controller = load_model()
    st.success("✅ Modelo cargado correctamente.")
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

# ─────────────────────────────────────────────
# Tabs: predicción individual y por lote
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Texto Individual", "📋 Clasificación por Lote"])

# ── TAB 1: Predicción individual ──────────────
with tab1:
    st.subheader("Clasificar un texto")

    example_texts = [
        "Selecciona un ejemplo...",
        "El acceso a la educación de calidad es fundamental para el desarrollo sostenible de las comunidades.",
        "Las energías renovables como la solar y la eólica son clave para combatir el cambio climático.",
        "Es necesario garantizar el acceso al agua potable y al saneamiento básico en zonas rurales.",
        "La igualdad de género implica garantizar los mismos derechos y oportunidades para hombres y mujeres.",
        "El crecimiento económico inclusivo debe generar empleo digno y proteger los derechos laborales.",
    ]

    selected_example = st.selectbox("💡 O elige un texto de ejemplo:", example_texts)

    default_text = "" if selected_example == "Selecciona un ejemplo..." else selected_example
    user_text = st.text_area(
        "✍️ Escribe o pega tu texto aquí:",
        value=default_text,
        height=150,
        placeholder="Ej: La educación de calidad es un derecho fundamental para todos los niños y niñas...",
    )

    if st.button("🔍 Clasificar", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Por favor ingresa un texto antes de clasificar.")
        else:
            with st.spinner("Clasificando..."):
                result = controller.predict(user_text)

            ods_num = result['ods_number']
            ods_info = ODS_INFO.get(ods_num, {"color": "#333333", "icon": "🌍"})
            ods_color = ods_info["color"]
                             
            col1, col2 = st.columns([1, 2])

            with col1:
                with col1:
                    st.markdown("### 🎯 Resultado")
                    ods_img = ods_info.get("img", "")
                    st.image(ods_img, width=250)

            with col2:
                st.markdown("### 📊 Top 5 Probabilidades")
                probs = result["probabilities"]
                top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                df_probs = pd.DataFrame(top5, columns=["ODS", "Probabilidad"])
                df_probs["ODS"] = df_probs["ODS"].apply(lambda x: f"ODS {x}")
                df_probs["Probabilidad (%)"] = (df_probs["Probabilidad"] * 100).round(2)

                fig = px.bar(
                    df_probs,
                    x="ODS",
                    y="Probabilidad (%)",
                    color="Probabilidad (%)",
                    color_continuous_scale="Blues",
                    text="Probabilidad (%)",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Clasificación por lote ─────────────
with tab2:
    st.subheader("Clasificar múltiples textos")
    st.markdown("Pega un texto por línea. El modelo clasificará cada uno de forma independiente.")

    batch_input = st.text_area(
        "📋 Textos (uno por línea):",
        height=200,
        placeholder="Texto 1\nTexto 2\nTexto 3\n...",
    )

    if st.button("🔍 Clasificar todos", type="primary", use_container_width=True):
        lines = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
        if not lines:
            st.warning("⚠️ Por favor ingresa al menos un texto.")
        else:
            with st.spinner(f"Clasificando {len(lines)} textos..."):
                results = controller.predict_batch(lines)

            df_results = pd.DataFrame([
                {
                    "Texto": text[:100] + "..." if len(text) > 100 else text,
                    "ODS Predicho": f"ODS {r['ods_number']}",
                    "Nombre del ODS": r["ods_name"],
                    "Confianza (%)": round(max(r["probabilities"].values()) * 100, 2),
                }
                for text, r in zip(lines, results)
            ])

            st.success(f"✅ {len(df_results)} textos clasificados.")
            st.dataframe(df_results, use_container_width=True)

            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar resultados como CSV",
                data=csv,
                file_name="clasificacion_ods.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Clasificador de ODS · Curso MAIA · Modelo: Regresión Logística + TF-IDF · F1-Score: 85.83%"
    "</div>",
    unsafe_allow_html=True,
)
