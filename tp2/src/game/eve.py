from tp2.src.game.characters import Character

class EVE:
    def evaluate(self, character: Character):
        """
        Evalúa el desempeño de un personaje utilizando la fórmula específica
        implementada en la clase correspondiente del personaje.
        
        :param character: Una instancia de la clase Character o sus derivados.
        :return: Un valor numérico que representa el desempeño del personaje.
        """
        # Llamar al método específico de la clase del personaje
        performance = character.calculate_performance()

        # Se podría aplicar una normalización u otros ajustes aquí si fuera necesario
        return performance
