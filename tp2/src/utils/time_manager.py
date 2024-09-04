
import time
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

    def time_is_up(self):
        """
        Verifica si el tiempo se ha agotado.
        
        :return: True si el tiempo ha expirado, False en caso contrario.
        """
        return self.time_remaining() <= 0