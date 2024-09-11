import json
import os
from genetic_algorithm.classes.hyperparameters import Hyperparameters

class ConfigLoader:
    def __init__(self, config_file, game_config_file, initial_population_file = None):
        self.config_file = config_file
        self.game_config_file = game_config_file
        self.initial_population = initial_population_file
        self.config = None

    def load(self)->Hyperparameters:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"El archivo de configuración {self.config_file} no existe.")

        with open(self.config_file, 'r') as file:
            try:
                self.config = json.load(file)
                ga_config = Hyperparameters(self.config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error al leer el archivo de configuración: {e}")

        return ga_config

    def load_game_config(self):
        if not os.path.exists(self.game_config_file):
            raise FileNotFoundError(f"El archivo de configuración del juego {self.game_config_file} no existe.")

        with open(self.game_config_file, 'r') as file:
            try:
                game_config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error al leer el archivo de configuración del juego: {e}")

        return game_config
    
    def load_initial_population(self):
        if not os.path.exists(self.initial_population):
            return None
        with open(self.initial_population, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error al leer el archivo de poblacion inicial: {e}")
            