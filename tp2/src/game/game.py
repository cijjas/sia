import random
import time
from tp2.src.game.characters import Warrior, Archer, Guardian, Mage, Character
from tp2.src.utils.attribute_alocator import AttributeAllocator, TimeManager
from eve import EVE
from utils.time_manager import TimeManager

def start_game(config, game_config):

    characters = game_config['characters_classes']
    attributes = game_config['attributes']
    points_range = game_config['points_range']
    height_range = game_config['height_range']
    time_limit_range = game_config['time_limit_range']

    time_limit = random.randint(time_limit_range[0], time_limit_range[1]) # seconds
    height = random.uniform(height_range[0], height_range[1])
    points = random.randint(points_range[0], points_range[1])

    time_manager = TimeManager(time_limit)

    return time_manager
    # start timer






