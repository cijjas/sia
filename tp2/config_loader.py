import json
import os

class ConfigLoader:
    def __init__(self, config_file):
        """
        Inicializa el cargador de configuración con la ruta del archivo JSON.
        
        :param config_file: Ruta al archivo de configuración JSON.
        """
        self.config_file = config_file
        self.config = None

    def load(self):
        """
        Carga la configuración desde el archivo JSON.
        
        :return: Diccionario con la configuración cargada.
        :raises FileNotFoundError: Si el archivo de configuración no existe.
        :raises ValueError: Si el archivo de configuración tiene un formato inválido.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"El archivo de configuración {self.config_file} no existe.")

        with open(self.config_file, 'r') as file:
            try:
                self.config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error al leer el archivo de configuración: {e}")

        self.validate_config()
        return self.config

    def validate_config(self):
        """
        Valida la configuración cargada para asegurarse de que contenga todos los campos necesarios.
        
        :raises ValueError: Si faltan campos obligatorios en la configuración o si los valores son inválidos.
        """
        required_fields = ["total_points", "time_limit", "height"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"El campo requerido '{field}' falta en la configuración.")

        # Validar que los valores sean correctos
        if not (100 <= self.config["total_points"] <= 200):
            raise ValueError("El 'total_points' debe estar entre 100 y 200.")

        if not (10 <= self.config["time_limit"] <= 120):
            raise ValueError("El 'time_limit' debe estar entre 10 y 120 segundos.")

        if not (1.3 <= self.config["height"] <= 2.0):
            raise ValueError("La 'height' debe estar entre 1.3m y 2.0m.")

