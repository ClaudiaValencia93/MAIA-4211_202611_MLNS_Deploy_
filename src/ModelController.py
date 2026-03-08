"""
ModelController.py
Módulo de control del modelo de clasificación de ODS
"""

import joblib
import os
import numpy as np
from typing import Union

# Nombres de los ODS para mostrar en la interfaz
ODS_NAMES = {
    1:  "ODS 1 - Fin de la Pobreza",
    2:  "ODS 2 - Hambre Cero",
    3:  "ODS 3 - Salud y Bienestar",
    4:  "ODS 4 - Educación de Calidad",
    5:  "ODS 5 - Igualdad de Género",
    6:  "ODS 6 - Agua Limpia y Saneamiento",
    7:  "ODS 7 - Energía Asequible y No Contaminante",
    8:  "ODS 8 - Trabajo Decente y Crecimiento Económico",
    9:  "ODS 9 - Industria, Innovación e Infraestructura",
    10: "ODS 10 - Reducción de las Desigualdades",
    11: "ODS 11 - Ciudades y Comunidades Sostenibles",
    12: "ODS 12 - Producción y Consumo Responsables",
    13: "ODS 13 - Acción por el Clima",
    14: "ODS 14 - Vida Submarina",
    15: "ODS 15 - Vida de Ecosistemas Terrestres",
    16: "ODS 16 - Paz, Justicia e Instituciones Sólidas",
    17: "ODS 17 - Alianzas para Lograr los Objetivos",
}


class ModelController:
    """
    Controlador del modelo de clasificación de ODS.
    Gestiona la carga del modelo y las predicciones.
    """

    def __init__(self, model_path: str):
        """
        Inicializa el controlador cargando el modelo desde disco.

        Args:
            model_path (str): Ruta al archivo .joblib del modelo guardado
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Carga el modelo desde el archivo .joblib.

        Returns:
            Pipeline de scikit-learn con el modelo entrenado

        Raises:
            FileNotFoundError: Si el archivo del modelo no existe
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en: {self.model_path}\n"
                "Asegúrate de haber guardado el modelo con joblib y colocarlo en la carpeta 'resources/'."
            )
        return joblib.load(self.model_path)

    def predict(self, text: str) -> dict:
        """
        Realiza la predicción del ODS para un texto dado.

        Args:
            text (str): Texto a clasificar

        Returns:
            dict con:
                - 'ods_number': número del ODS predicho (int)
                - 'ods_name': nombre completo del ODS (str)
                - 'probabilities': dict con probabilidades de cada clase
        """
        if not text or not text.strip():
            raise ValueError("El texto de entrada no puede estar vacío.")

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        prob_dict = {
            int(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

        ods_number = int(prediction)
        ods_name = ODS_NAMES.get(ods_number, f"ODS {ods_number}")

        return {
            "ods_number": ods_number,
            "ods_name": ods_name,
            "probabilities": prob_dict,
        }

    def predict_batch(self, texts: list) -> list:
        """
        Realiza predicciones para una lista de textos.

        Args:
            texts (list): Lista de textos a clasificar

        Returns:
            list de dicts con resultados de predict() para cada texto
        """
        return [self.predict(text) for text in texts]
