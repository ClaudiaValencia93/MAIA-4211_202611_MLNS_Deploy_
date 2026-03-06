"""
save_model.py
─────────────────────────────────────────────────────────────────────────────
Ejecuta este script AL FINAL de tu notebook (o como celda adicional) para
guardar el modelo entrenado en la carpeta resources/.

INSTRUCCIONES:
1. Asegúrate de haber ejecutado todas las celdas del notebook
   (entrenamiento con GridSearchCV completado).
2. Agrega una celda al final del notebook con el contenido de este archivo,
   O copia y pega el bloque de código en una nueva celda.
3. Ejecuta esa celda. Se generará el archivo: resources/model.joblib
4. Ese archivo es el único que necesitas copiar a la carpeta resources/ del proyecto.
─────────────────────────────────────────────────────────────────────────────
"""

import joblib
import os

# Crear carpeta resources si no existe
os.makedirs("resources", exist_ok=True)

# Guardar el mejor modelo encontrado por GridSearchCV
joblib.dump(grid_search.best_estimator_, "resources/model.joblib")

print("Modelo guardado en: resources/model.joblib")
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"F1-Score (cv):      {grid_search.best_score_:.4f}")
