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
    1:  {"color": "#E5243B", "img": "img/ods1.png"},
    2:  {"color": "#DDA63A", "img": "img/ods2.png"},
    3:  {"color": "#4C9F38", "img": "img/ods3.png"},
    4:  {"color": "#C5192D", "img": "img/ods4.png"},
    5:  {"color": "#FF3A21", "img": "img/ods5.png"},
    6:  {"color": "#26BDE2", "img": "img/ods6.png"},
    7:  {"color": "#FCC30B", "img": "img/ods7.png"},
    8:  {"color": "#A21942", "img": "img/ods8.png"},
    9:  {"color": "#FD6925", "img": "img/ods9.png"},
    10: {"color": "#DD1367", "img": "img/ods10.png"},
    11: {"color": "#FD9D24", "img": "img/ods11.png"},
    12: {"color": "#BF8B2E", "img": "img/ods12.png"},
    13: {"color": "#3F7E44", "img": "img/ods13.png"},
    14: {"color": "#0A97D9", "img": "img/ods14.png"},
    15: {"color": "#56C02B", "img": "img/ods15.png"},
    16: {"color": "#00689D", "img": "img/ods16.png"},
    17: {"color": "#19486A", "img": "img/ods17.png"},
}

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de texto de ODS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        /* Pestañas hover y activa */
        .stTabs [data-baseweb="tab"]:hover {
            color: #0078B4 !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #0078B4 !important;
            border-bottom-color: #0078B4 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #0078B4 !important;
        }

        /* Todos los inputs, selectbox y textarea - quitar rojo */
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="base-input"]:focus-within,
        div[data-baseweb="base-input"]:hover,
        div[data-baseweb="textarea"]:focus-within,
        div[data-baseweb="textarea"]:hover,
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: #0078B4 !important;
            outline-color: #0078B4 !important;
            box-shadow: 0 0 0 1px #0078B4 !important;
        }

        /* Dropdown items hover */
        li[role="option"]:hover {
            background-color: #0078B4 !important;
            color: white !important;
        }

        /* Botón primario */
        .stButton > button[kind="primary"] {
            background-color: #0078B4 !important;
            border-color: #0078B4 !important;
            color: white !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #005a8e !important;
            border-color: #005a8e !important;
        }
        /* Botón secundario */
        .stButton > button[kind="secondary"] {
            background-color: #0078B4 !important;
            border-color: #0078B4 !important;
            color: white !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: #005a8e !important;
            border-color: #005a8e !important;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Carga del modelo cacheado para eficiencia
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resources", "model.joblib")

@st.cache_resource
def load_model():
    return ModelController(MODEL_PATH)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    # st.image(os.path.join(os.path.dirname(__file__), "img", "poster_ods_.png"), use_container_width=True)
    col_logo1, col_logo2 = st.columns(2)
    with col_logo1:
        st.image(os.path.join(os.path.dirname(__file__), "img", "logo_ods.png"), width=600)
    with col_logo2:
        st.image(os.path.join(os.path.dirname(__file__), "img", "logo_uniandes.png"), width=120)
    st.markdown("## Clasificador de texto de los ODS")
    st.markdown(
        "En 2015, la ONU aprobó la Agenda 2030 con **17 Objetivos de Desarrollo Sostenible (ODS)** para mejorar la vida"
        " de todas las personas sin dejar a nadie atrás, abordando desafíos como la pobreza, la educación," 
        " la salud, el empleo y el cambio climático. Esta aplicación utiliza un modelo de Machine Learning"
        " basado en Regresión Logística y TF-IDF para clasificar automáticamente textos según el ODS al que" 
        " pertenecen, contribuyendo a visibilizar y conectar el conocimiento con los objetivos globales" 
        " de desarrollo sostenible"
    )
    st.markdown("---")
    st.markdown("**Modelo:** Regresión Logística + TF-IDF")
    st.markdown("**Métrica F1-Score:** 85.83%")
    st.markdown("---")
    st.markdown("**Autores:**")
    st.markdown("Claudia Valencia Morales")
    st.markdown("Sandro Fabián Castro")
    st.markdown("---")
    st.markdown("Maestría en Inteligencia Artificial Universidad de los Andes")

# ─────────────────────────────────────────────
# Título principal
# ─────────────────────────────────────────────
col_izq, col_centro, col_der = st.columns([1, 2, 1])
with col_centro:
     st.markdown(
        """
        <div style="margin-top: -100px;">
        """,
        unsafe_allow_html=True,
    )
     st.image(os.path.join(os.path.dirname(__file__), "img", "poster_ods_.png"), width=250)
     st.markdown(
        """
        <div style="
            text-align: center;
            color: #009EDB;
            font-size: 1.5rem;
            font-weight: 900;
            letter-spacing: 3px;
            font-family: 'Arial Black', sans-serif;
            margin-top: 10px;
        ">
            Clasificador de Textos
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("Ingresa un texto relacionado con los Objetivos de Desarrollo Sostenible y el modelo predecirá a cuál ODS pertenece")

# ─────────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────────
try:
    controller = load_model()
    st.markdown(
        """
        <div style="
            background-color: #0078B4;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
        ">
            ✔️ Modelo cargado correctamente
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

# ─────────────────────────────────────────────
# Tabs: predicción individual y por lote
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Texto individual", "📋 Clasificación por lote"])

# ── TAB 1: Predicción individual ──────────────
with tab1:
    st.subheader("Clasificar un texto")
    example_texts = [
        "Selecciona un ejemplo...",
        "El acceso a la educación de calidad es fundamental para el desarrollo sostenible de las comunidades",
        "Las energías renovables como la solar y la eólica son clave para combatir el cambio climático",
        "Es necesario garantizar el acceso al agua potable y al saneamiento básico en zonas rurales",
        "La igualdad de género implica garantizar los mismos derechos y oportunidades para hombres y mujeres",
        "El crecimiento económico inclusivo debe generar empleo digno y proteger los derechos laborales",
    ]
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    if "last_example" not in st.session_state:
        st.session_state.last_example = "Selecciona un ejemplo..."

    def on_example_change():
        sel = st.session_state["selectbox_ejemplo"]
        if sel != "Selecciona un ejemplo...":
            st.session_state.user_input = sel
            st.session_state.input_key += 1
        st.session_state.last_example = sel

    st.selectbox(
        "💡 Elige un texto de ejemplo:",
        example_texts,
        key="selectbox_ejemplo",
        on_change=on_example_change
    )

    user_text = st.text_area(
        "✍️ O escribe y/o pega tu texto aquí:",
        value=st.session_state.user_input,
        height=150,
        key=f"texto_{st.session_state.input_key}",
        placeholder="Ej: La educación de calidad es un derecho fundamental para todos los niños y niñas...",
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        clasificar = st.button("🔍 Clasificar el texto", type="primary", use_container_width=True, key="btn_individual")
    with col_btn2:
        if st.button("🗑️ Borrar texto", use_container_width=True, key="btn_limpiar1"):
            st.session_state.user_input = ""
            st.session_state.input_key += 1
            st.rerun()

    if clasificar:
        if not user_text.strip():
            st.warning("⚠️ Por favor ingresa un texto antes de clasificar")
        else:
            palabras_validas = [w for w in user_text.strip().split() if w.isalpha() and len(w) > 3]
            if len(palabras_validas) < 5:
                st.warning("⚠️ Por favor ingresa un texto válido para clasificar")
            else:
                with st.spinner("Clasificando..."):
                    result = controller.predict(user_text)
                ods_num = result['ods_number']
                ods_info = ODS_INFO.get(ods_num, {"color": "#333333", "img": ""})
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### 🎯 Resultado del ODS")
                    ods_img = ods_info.get("img", "")
                    st.image(ods_img, width=250)
                with col2:
                    st.markdown("### 📊 Top 5 de probabilidades")
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
                    st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["zoom2d", "pan2d",
                    "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]})

# with tab1:
#     st.subheader("Clasificar un texto")

#     example_texts = [
#         "Selecciona un ejemplo...",
#         "El acceso a la educación de calidad es fundamental para el desarrollo sostenible de las comunidades",
#         "Las energías renovables como la solar y la eólica son clave para combatir el cambio climático",
#         "Es necesario garantizar el acceso al agua potable y al saneamiento básico en zonas rurales",
#         "La igualdad de género implica garantizar los mismos derechos y oportunidades para hombres y mujeres",
#         "El crecimiento económico inclusivo debe generar empleo digno y proteger los derechos laborales",
#     ]

#     selected_example = st.selectbox("💡 Elige un texto de ejemplo:", example_texts)

#     default_text = "" if selected_example == "Selecciona un ejemplo..." else selected_example
#     if "texto" not in st.session_state:
#         st.session_state.texto = default_text

#     user_text = st.text_area(
#         "✍️ O escribe y/o pega tu texto aquí:",
#         height=150,
#         key="texto",
#         placeholder="Ej: La educación de calidad es un derecho fundamental para todos los niños y niñas...",
#     )

#     col_btn1, col_btn2 = st.columns([3, 1])
#     with col_btn1:
#         clasificar = st.button("🔍 Clasificar el texto", type="primary", use_container_width=True, key="btn_individual")
#     with col_btn2:
#         if st.button("🗑️ Borrar texto", use_container_width=True, key="btn_limpiar1"):
#             del st.session_state["texto"]
#             st.rerun()

#     if clasificar:
#         if not user_text.strip():
#             st.warning("⚠️ Por favor ingresa un texto antes de clasificar")
#         else:
#             palabras_validas = [w for w in user_text.strip().split() if w.isalpha() and len(w) > 3]
#             if len(palabras_validas) < 5:
#                 st.warning("⚠️ Por favor ingresa un texto válido para clasificar")
#             else:
#                 with st.spinner("Clasificando..."):
#                     result = controller.predict(user_text)
#                 ods_num = result['ods_number']
#                 ods_info = ODS_INFO.get(ods_num, {"color": "#333333", "icon": "🌍"})
                                         
#             col1, col2 = st.columns([1, 2])

#             with col1:
#                 with col1:
#                    st.markdown("### 🎯 Resultado del ODS")
#                    ods_img = ods_info.get("img", "")
#                    st.image(ods_img, width=250)

#             with col2:
#                 st.markdown("### 📊 Top 5 de probabilidades")
#                 probs = result["probabilities"]
#                 top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
#                 df_probs = pd.DataFrame(top5, columns=["ODS", "Probabilidad"])
#                 df_probs["ODS"] = df_probs["ODS"].apply(lambda x: f"ODS {x}")
#                 df_probs["Probabilidad (%)"] = (df_probs["Probabilidad"] * 100).round(2)

#                 fig = px.bar(
#                     df_probs,
#                     x="ODS",
#                     y="Probabilidad (%)",
#                     color="Probabilidad (%)",
#                     color_continuous_scale="Blues",
#                     text="Probabilidad (%)",
#                 )
#                 fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
#                 fig.update_layout(showlegend=False, height=300)
#                 st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["zoom2d", "pan2d", 
#                 "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]
#                 })

# ── TAB 2: Clasificación por lote ─────────────
with tab2:
    st.subheader("Clasificar múltiples textos")
    st.markdown("Pega un texto por línea. El modelo clasificará cada uno de forma independiente")

    if "batch" not in st.session_state:
        st.session_state.batch = ""

    batch_input = st.text_area(
        "📋 Textos (uno por línea):",
        height=200,
        key="batch",
        placeholder="Texto 1\nTexto 2\nTexto 3\n...",
    )

    col_btn3, col_btn4 = st.columns([3, 1])
    with col_btn3:
        clasificar_lote = st.button("🔍 Clasificar los textos", type="primary", use_container_width=True, key="btn_lote")
    with col_btn4:
        if st.button("🗑️ Borrar texto", use_container_width=True, key="btn_limpiar2"):
           del st.session_state["batch"]
           st.rerun()

    if clasificar_lote:
            lines = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
            if not lines:
                st.warning("⚠️ Por favor ingresa al menos un texto antes de clasificar")
            else:
                lines_validas = [l for l in lines if len([w for w in l.split() if w.isalpha() and len(w) > 3]) >= 5]
                if not lines_validas:
                    st.warning("⚠️ Por favor ingresa textos válidos para clasificar")
                else:
                    with st.spinner(f"Clasificando {len(lines_validas)} textos..."):
                        results = controller.predict_batch(lines_validas)
                    df_results = pd.DataFrame([
                        {
                            "Texto": text[:100] + "..." if len(text) > 100 else text,
                            "ODS Predicho": f"ODS {r['ods_number']}",
                            "Nombre del ODS": r["ods_name"],
                            "Confianza (%)": round(max(r["probabilities"].values()) * 100, 2),
                        }
                        for text, r in zip(lines_validas, results)
                    ])
                    st.success(f"✔️ {len(df_results)} textos clasificados.")
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
    "Clasificador de textos de los ODS · Curso MLNS MAIA - Uniandes· Modelo: Regresión Logística + TF-IDF · F1-Score: 85.83%"
    "</div>",
    unsafe_allow_html=True,
)
