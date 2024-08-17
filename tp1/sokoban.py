import pygame
from core.state import State
from core.node import Node
from core.map_parser import parse_map
from core.search_engine import search
from core.heuristics import *
from algorithms.bfs import bfs
import sys

TILE_SIZE = 64
direction_to_key = {
    (0, 1): pygame.K_DOWN,   # Moving down
    (1, 0): pygame.K_RIGHT,  # Moving right
    (0, -1): pygame.K_UP,    # Moving up
    (-1, 0): pygame.K_LEFT   # Moving left
}

def load_images(tile_size):
    images = {
        '#': pygame.image.load('core/resources/wall.jpg').convert_alpha(),
        '@': pygame.image.load('core/resources/player.jpg').convert_alpha(),
        '$': pygame.image.load('core/resources/box.jpeg').convert_alpha(),
        ' ': pygame.image.load('core/resources/empty.jpg').convert_alpha(),
        '.': pygame.image.load('core/resources/goal.png').convert_alpha(),
        '*': pygame.image.load('core/resources/box_on_goal.png').convert_alpha(),
        'p': pygame.image.load('core/resources/player.jpg').convert_alpha()
    }

    for key in images:
        images[key] = pygame.transform.scale(images[key], (tile_size, tile_size))
    return images

def draw_board(screen, map_data, images, tile_size, map_width, map_height):
    for x in range(0, map_width * tile_size, tile_size):
        for y in range(0, map_height * tile_size, tile_size):
            screen.blit(images[' '], (x, y))

    walls = map_data.walls
    goals = map_data.goals
    boxes = map_data.boxes
    player = map_data.player
    for x, y in walls:
        screen.blit(images['#'], (x * tile_size, y * tile_size))
    for x, y in goals:
        screen.blit(images['.'], (x * tile_size, y * tile_size))
    for x, y in boxes:
        screen.blit(images['$'], (x * tile_size, y * tile_size))
    for x, y in goals & boxes:
        screen.blit(images['*'], (x * tile_size, y * tile_size))
    for x, y in goals & {player}:
        screen.blit(images['p'], (x * tile_size, y * tile_size))
    screen.blit(images['@'], (player[0] * tile_size, player[1] * tile_size))

def show_action_sequence(action_sequence, game_state, map_data):
    pygame.init()
    map_width = map_data['width'] * TILE_SIZE
    map_height = map_data['height'] * TILE_SIZE

    screen = pygame.display.set_mode((map_width, map_height))

    images = load_images(TILE_SIZE)
    running = True
    path_index = 0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in direction_to_key.values():
                    # Assuming the direction is mapped directly to the event.key in direction_to_key
                    for direction, key in direction_to_key.items():
                        if key == event.key:
                            new_state = game_state.move_player(*direction)
                            if new_state:
                                game_state = new_state
                                break
                    draw_board(screen, game_state, images, TILE_SIZE, map_width, map_height)
                    pygame.display.flip()

        if path_index < len(action_sequence):
            direction = action_sequence[path_index]
            key_event = pygame.event.Event(pygame.KEYDOWN, key=direction_to_key[direction])
            pygame.event.post(key_event)
            path_index += 1
            pygame.time.delay(500)  # Add delay to see each move

        clock.tick(60)

    pygame.quit()


    return

def main():
    if (len(sys.argv) != 2):
        print("Choose a map!")
        sys.exit(1)

    # Initial Values
    map_data = parse_map(sys.argv[1])
    initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'])
    initial_node = Node(initial_state, None, 0, 0)

    # Use Search Algorithm
    search_result = bfs(initial_node)

    # Results
    print(search_result)
    show_action_sequence(search_result, initial_state, map_data)

if __name__ == "__main__":
    main()
