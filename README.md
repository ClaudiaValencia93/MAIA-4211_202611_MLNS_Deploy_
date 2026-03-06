# 🌍 Clasificador de Textos por ODS

Aplicación Streamlit para clasificar textos según los **16 Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

**Modelo:** Regresión Logística + TF-IDF + TruncatedSVD  
**F1-Score en test:** 85.83%

---

## 📁 Estructura del proyecto

```
ODS_Classifier/
│
├── streamlit_app.py          # Aplicación principal de Streamlit
├── requirements.txt          # Dependencias de Python
├── save_model.py             # Script para guardar el modelo desde el notebook
│
├── src/
│   ├── DataPreprocessing.py  # Función text_preprocess (tokenización, stopwords, lematización)
│   └── ModelController.py    # Clase ModelController (carga y predicción)
│
└── resources/
    └── model.joblib          
```

---

## Pasos para el deployment

### Paso 1 — Guardar el modelo desde tu notebook

1. Abre tu notebook `3_Solucion_microproyecto_dos.ipynb` en Google Colab o Jupyter.
2. Ejecuta **todas las celdas** hasta que el `GridSearchCV` termine el entrenamiento.
3. Agrega una **nueva celda al final** con este código y ejecútala:

```python
import joblib
import os

os.makedirs("resources", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "resources/model.joblib")
print("Modelo guardado en: resources/model.joblib")
```

4. Descarga el archivo `model.joblib` generado.
5. Colócalo en la carpeta `resources/` de este proyecto.

---

### Paso 2 — Subir el proyecto a GitHub

1. Crea un repositorio nuevo en [github.com](https://github.com) (puede ser público o privado).
2. Sube **todos los archivos** de esta carpeta al repositorio:
   - `streamlit_app.py`
   - `requirements.txt`
   - `src/DataPreprocessing.py`
   - `src/ModelController.py`
   - `resources/model.joblib` (el que generaste en el Paso 1)

   Puedes hacerlo arrastrando los archivos directamente en la interfaz web de GitHub,
   o usando Git desde la terminal:

```bash
git init
git add .
git commit -m "Primer commit: clasificador ODS"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

---

### Paso 3 — Desplegar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en **"New app"**.
3. Completa los campos:
   - **Repository:** `TU_USUARIO/TU_REPOSITORIO`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Haz clic en **"Deploy!"**.
5. Espera ~2 minutos mientras se instalan las dependencias.
6. ¡Tu app estará disponible en una URL pública! 🎉

---

## Nota sobre el archivo model.joblib

El archivo `model.joblib` puede pesar varios MB. GitHub tiene un límite de **100 MB por archivo**.
Si tu modelo supera ese límite, usa [Git LFS](https://git-lfs.github.com/):

```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git add resources/model.joblib
git commit -m "Agregar modelo con LFS"
git push
```

---

## Probar localmente (opcional)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
