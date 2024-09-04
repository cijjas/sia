import random
import time
from characters import Warrior, Archer, Guardian, Mage, Character
from utils import AttributeAllocator, TimeManager
from eve import EVE

def start(config):
    """
    Inicializa el juego y coordina todos los componentes. Permite al jugador ajustar
    los atributos del personaje dentro de un tiempo limitado y muestra el mejor personaje
    al final del tiempo.

    :param config: Diccionario de configuración ya validado.
    """
    # Definir los tipos de personajes
    character_classes = [Warrior, Archer, Guardian, Mage]

    # Extraer variables desde la configuración
    total_points = config['total_points']
    time_limit = config['time_limit']
    character_height = config['height']

    # Seleccionar una clase de personaje aleatoriamente
    character_class = random.choice(character_classes)
    character: Character = character_class(character_height)

    # Configurar el asignador de atributos y el administrador de tiempo
    attribute_allocator = AttributeAllocator(total_points)
    time_manager = TimeManager(time_limit)
    
    character.set_attributes(attribute_allocator.random_allocate())

    # Crear instancia del EVE
    eve = EVE()
    
    # Mostrar mensaje inicial
    print(f"Class selected: {character_class.__name__}")
    print(f"Adjust your character within {time_limit} seconds.")
    
    # Temporizador para que el jugador ajuste los atributos
    time_manager.start()
    best_attributes = None
    best_performance = float('-inf')

    while time_manager.time_remaining() > 0:
        # Solicitar al jugador que ingrese los atributos
        print("Enter your attributes:")
        strength = int(input("Strength: "))
        dexterity = int(input("Dexterity: "))
        intelligence = int(input("Intelligence: "))
        vigor = int(input("Vigor: "))
        constitution = int(input("Constitution: "))
        
        # Asignar los atributos al personaje
        attributes = {
            'strength': strength,
            'dexterity': dexterity,
            'intelligence': intelligence,
            'vigor': vigor,
            'constitution': constitution
        }
        
        if sum(attributes.values()) != total_points:
            print(f"Error: The total points must be exactly {total_points}.")
            continue
        
        character.set_attributes(attributes)
        character.set_height(character_height)
        
        # Calcular el desempeño usando EVE
        performance = eve.evaluate(character)
        
        if performance > best_performance:
            best_performance = performance
            best_attributes = attributes
        
        # Mostrar el desempeño actual
        print(f"Current performance: {performance}")
    
    # Terminado el tiempo, mostrar el mejor personaje
    character.set_attributes(best_attributes)
    character.set_height(character_height)
    
    print("\nTime's up!")
    print(f"Best character type: {character_class.__name__}")
    print(f"Final Attributes: {best_attributes}")
    print(f"Height: {character_height:.2f} meters")
    print(f"Performance: {best_performance}")

