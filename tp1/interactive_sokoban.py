import pygame
from core.state import State
from core.node import Node
from core.map_parser import parse_map
from core.heuristics import *
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.greedy import global_greedy
from algorithms.greedy import local_greedy
from algorithms.a_star import a_star
from algorithms.iddfs import iddfs
import sys

TILE_SIZE = 40
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
        '+': pygame.image.load('core/resources/player.jpg').convert_alpha()
    }

    for key in images:
        images[key] = pygame.transform.scale(images[key], (tile_size, tile_size))
    return images


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
            pygame.time.delay(1)

        clock.tick(1000)

    pygame.quit()

    #   saerch_result, expanded_nodes = dfs(initial_node)#         
    return


def main_menu(screen, font, map_name, game_state, images, map_data, algorithm_finished):
    options = ['BFS', 'DFS', 'Global Greedy', 'Local Greedy', 'A*', 'IDDFS']
    buttons = {}
    menu_width = 300  # Width reserved for the menu

    # Position algorithm buttons only if algorithm has not finished
    if not algorithm_finished:
        for i, option in enumerate(options):
            buttons[option] = (50, 50 + 40 * i, 200, 35)

    separator_y = 50 + 40 * len(options) + 20  # Position for the separator line

    # Only show 'Reset' button if the algorithm has finished
    if algorithm_finished:
        buttons['Reset'] = (50, separator_y + 20, 200, 35)

    while True:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for key, (bx, by, bw, bh) in buttons.items():
                    if bx <= mouse_x <= bx + bw and by <= mouse_y <= by + bh:
                        return key  # Return the clicked algorithm or 'Reset'

        screen.fill((6, 20, 6))  # Clear screen
        pygame.draw.line(screen, (100, 100, 100), (menu_width, 0), (menu_width, 768), 1)  # Draw vertical line
        
        text = font.render(map_name, True, (255, 255, 255))
        screen.blit(text, (50, 710))

        pygame.draw.line(screen, (255, 255, 255), (40, separator_y - 10), (menu_width - 50, separator_y - 10), 1)  # Draw separator line

        for key, (bx, by, bw, bh) in buttons.items():
            text_color = (94, 242, 122) if bx <= mouse_x <= bx + bw and by <= mouse_y <= by + bh else (255, 255, 255)
            text = font.render(key, True, text_color)
            screen.blit(text, (bx, by))

        draw_board(screen, game_state, images, TILE_SIZE, 350, 150, map_data)  # Update game state view
        pygame.display.flip()

        # Wait a bit to reduce CPU usage in a busy loop
        pygame.time.wait(100)


def action_sequence_control(screen, initial_state, images, map_data, is_running, is_paused):
    global game_state
    game_state = initial_state
    while is_running:
        if not is_paused:
            # Proceed with showing the action sequence or updating the game state
            show_action_sequence(screen, game_state, images, map_data)
        else:
            # Draw the current state of the board, but do not proceed in the sequence
            draw_board(screen, game_state, images, TILE_SIZE, 350, 150, map_data)


def show_action_sequence(action_sequence, game_state, images, map_data, screen, x_offset, y_offset):
    path_index = 0
    while path_index < len(action_sequence):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Assuming each action in the action_sequence is a tuple (dx, dy)
        dx, dy = action_sequence[path_index]  # Unpack the tuple into dx and dy
        game_state = game_state.move_player(dx, dy)  # Correctly pass both dx and dy

        draw_board(screen, game_state, images, TILE_SIZE, x_offset, y_offset, map_data)
        path_index += 1
        pygame.time.wait(100)  # Slow down the animation for visibility
        pygame.display.flip()

def draw_board(screen, game_state, images, tile_size, x_offset, y_offset, map_data):
    for y in range(map_data['height']):
        for x in range(map_data['width']):
            screen.blit(images[' '], (x * tile_size + x_offset, y * tile_size + y_offset))
    for x, y in game_state.walls:
        screen.blit(images['#'], (x * tile_size + x_offset, y * tile_size + y_offset))
    for x, y in game_state.goals:
        screen.blit(images['.'], (x * tile_size + x_offset, y * tile_size + y_offset))
    for x, y in game_state.boxes:
        screen.blit(images['$'], (x * tile_size + x_offset, y * tile_size + y_offset))
    for x, y in game_state.goals & game_state.boxes:
        screen.blit(images['*'], (x * tile_size + x_offset, y * tile_size + y_offset))
    for x, y in game_state.goals & {game_state.player}:
        screen.blit(images['+'], (x * tile_size + x_offset, y * tile_size + y_offset))
    screen.blit(images['@'], (game_state.player[0] * tile_size + x_offset, game_state.player[1] * tile_size + y_offset))
    

def run_algorithm(algorithm, initial_node):
    if algorithm == 'BFS':
        return bfs(initial_node)
    elif algorithm == 'DFS':
        return dfs(initial_node)
    elif algorithm == 'Global Greedy':
        return global_greedy(initial_node, DeadlockCorner())
    elif algorithm == 'Local Greedy':
        return local_greedy(initial_node,DeadlockCorner())
    elif algorithm == 'A*':
        return a_star(initial_node, DeadlockCorner())
    elif algorithm == 'IDDFS':
        return iddfs(initial_node)

def main():
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))
    font = pygame.font.Font("core/resources/fonts/NeueHaasDisplayMediu.ttf", 24)
    window_title = "Sokoban"
    pygame.display.set_caption(window_title)
    input_map = sys.argv[1]
    map_name = input_map.split('/')[-1].split('.')[0]
    map_data = parse_map(input_map)
    images = load_images(TILE_SIZE)
    initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'])

    algorithm_finished = False
    current_state = initial_state

    while True:
        choice = main_menu(screen, font, map_name, current_state, images, map_data, algorithm_finished)
        if choice in ['BFS', 'DFS', 'Global Greedy', 'Local Greedy', 'A*', 'IDDFS']:
            initial_node = Node(current_state, None, None, 0)
            search_result, _ = run_algorithm(choice, initial_node)
            # Execute all actions returned by the algorithm
            for action in search_result:
                dx, dy = action
                new_state = current_state.move_player(dx, dy)
                if new_state:
                    current_state = new_state
                    draw_board(screen, current_state, images, TILE_SIZE, 350, 150, map_data)
                    pygame.display.flip()
                pygame.time.wait(100)
            algorithm_finished = True  # Set this flag once the algorithm is done
        elif choice == 'Reset':
            current_state = initial_state
            algorithm_finished = False  # Reset the flag so algorithms can run again


if __name__ == "__main__":
    main()
