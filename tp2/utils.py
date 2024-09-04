import time
from collections import defaultdict
import random

class AttributeAllocator:
    def __init__(self, total_points):
        """
        Inicializa el asignador de atributos con un total de puntos a distribuir.
        
        :param total_points: Número total de puntos disponibles para asignar a los atributos.
        """
        self.total_points = total_points
        self.allocated_points = defaultdict(int)

    def allocate(self, strength=0, dexterity=0, intelligence=0, vigor=0, constitution=0):
        """
        Asigna puntos a los atributos según los valores proporcionados.
        
        :param strength: Puntos a asignar a la fuerza.
        :param dexterity: Puntos a asignar a la destreza.
        :param intelligence: Puntos a asignar a la inteligencia.
        :param vigor: Puntos a asignar al vigor.
        :param constitution: Puntos a asignar a la constitución.
        :return: Un diccionario con los puntos asignados a cada atributo.
        :raises ValueError: Si la suma de los puntos asignados excede el total disponible.
        """
        points_allocated = strength + dexterity + intelligence + vigor + constitution

        if points_allocated > self.total_points:
            raise ValueError("Los puntos asignados superan el total disponible.")

        self.allocated_points['strength'] = strength
        self.allocated_points['dexterity'] = dexterity
        self.allocated_points['intelligence'] = intelligence
        self.allocated_points['vigor'] = vigor
        self.allocated_points['constitution'] = constitution

        return self.allocated_points

    def random_allocate(self):
        """
        Asigna puntos aleatorios a los atributos dentro del rango permitido.
        
        :return: Un diccionario con los puntos asignados a cada atributo.
        """
        points_remaining = self.total_points
        for attribute in ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution']:
            max_points = min(points_remaining, 100)
            points = random.randint(0, max_points)
            self.allocated_points[attribute] = points
            points_remaining -= points

        return self.allocated_points

    def remaining_points(self):
        """
        Devuelve la cantidad de puntos restantes que aún no han sido asignados.
        
        :return: Número de puntos restantes.
        """
        return self.total_points - sum(self.allocated_points.values())

class TimeManager:
    def __init__(self, time_limit):
        """
        Inicializa el gestor de tiempo con un límite de tiempo dado.
        
        :param time_limit: Tiempo límite en segundos para completar la asignación de atributos.
        """
        self.time_limit = time_limit
        self.start_time = None

    def start(self):
        """Inicia el temporizador."""
        self.start_time = time.time()

    def time_remaining(self):
        """
        Calcula el tiempo restante.
        
        :return: Tiempo restante en segundos.
        """
        elapsed_time = time.time() - self.start_time
        return max(self.time_limit - elapsed_time, 0)

    def is_time_up(self):
        """
        Verifica si el tiempo se ha agotado.
        
        :return: True si el tiempo ha expirado, False en caso contrario.
        """
        return self.time_remaining() <= 0
