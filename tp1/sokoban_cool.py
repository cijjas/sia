import pygame
from core.structure.state import State
from core.structure.node import Node
from core.utils.map_parser import parse_map
from core.heuristics import *
from core.algorithms.a_star import a_star
from core.algorithms.bfs import bfs
from core.algorithms.dfs import dfs
from core.algorithms.greedy import local_greedy, global_greedy
from core.algorithms.iddfs import iddfs
import time
import sys
import os



TILE_SIZE = 40
direction_to_key = {
    (0, 1): pygame.K_DOWN,   # Moving down
    (1, 0): pygame.K_RIGHT,  # Moving right
    (0, -1): pygame.K_UP,    # Moving up
    (-1, 0): pygame.K_LEFT   # Moving left
}



def load_images(tile_size, texture_pack='default'):
    tiles = {
        '#': 'wall',
        '@': 'player',
        '$': 'box',
        ' ': 'empty',
        '.': 'goal',
        '*': 'box_on_goal',
        '+': 'player'
    }
    
    images = {}
    for key, value in tiles.items():
        for extension in ['png', 'jpg', 'jpeg', 'gif']:
            file_path = f'resources/texture_packs/{texture_pack}/{value}.{extension}'
            if os.path.exists(file_path):
                images[key] = pygame.image.load(file_path).convert_alpha()
                break

    for key in images:
        images[key] = pygame.transform.scale(images[key], (tile_size, tile_size))

    return images

def render_left_justified_textbox(screen, text, pos, font, color, max_width):
    words = text.split(' ')
    x, y = pos
    space = font.size(' ')[0]  # Width of a space.
    current_line = []
    current_width = 0

    lines = []
    # Split words into lines
    for word in words:
        word_width = font.size(word)[0]
        if current_width + word_width >= max_width:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width + space
        else:
            current_line.append(word)
            current_width += word_width + space

    if current_line:
        lines.append(' '.join(current_line))

    # Render each line left-justified
    current_y = y
    for line in lines:
        line_surface = font.render(line, True, color)
        screen.blit(line_surface, (x, current_y))
        current_y += font.get_height()



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
        return global_greedy(initial_node, [DeadlockCorner()])
    elif algorithm == 'Local Greedy':
        return local_greedy(initial_node, [DeadlockCorner()])
    elif algorithm == 'A*':
        return a_star(initial_node, [DeadlockCorner()])
    elif algorithm == 'IDDFS':
        return iddfs(initial_node)

def main_menu(screen, font, hover_font, map_name, game_state, images, map_data, algorithm_finished):
    options = ['BFS', 'DFS', 'Global Greedy', 'Local Greedy', 'A*', 'IDDFS']
    buttons = {}
    menu_width = 300  # Width reserved for the menu

    if not algorithm_finished:
        for i, option in enumerate(options):
            buttons[option] = (50, 50 + 40 * i, 200, 35)

    separator_y = 50 + 40 * len(options) + 20

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

        screen.fill((0,0,0))
        pygame.draw.line(screen, (100, 100, 100), (menu_width, 0), (menu_width, 768), 1)

        text = font.render(map_name, True, (255, 255, 255))
        screen.blit(text, (50, 710))

        pygame.draw.line(screen, (255, 255, 255), (40, separator_y - 10), (menu_width - 50, separator_y - 10), 1)

        for key, (bx, by, bw, bh) in buttons.items():
            text_color = (94, 242, 122) if bx <= mouse_x <= bx + bw and by <= mouse_y <= by + bh else (255, 255, 255)
            text = font.render(key, True, text_color)
            screen.blit(text, (bx, by))

        draw_board(screen, game_state, images, TILE_SIZE, 350, 150, map_data)
        pygame.display.flip()

        pygame.time.wait(100)

def render_textbox(screen, text, pos, font, color, max_width):
    words = text.split(' ')
    x, y = pos
    space = font.size(' ')[0]  # Width of a space.
    current_line = []
    current_width = 0

    for word in words:
        word_width = font.size(word)[0]
        if current_width + word_width >= max_width:
            # Render the current line then start a new one
            line_surface = font.render(' '.join(current_line), True, color)
            # Centering the text:
            screen.blit(line_surface, (x + (max_width - line_surface.get_width()) // 2, y))
            y += font.get_height()  # Move y to start a new line
            current_line = [word]
            current_width = word_width + space
        else:
            current_line.append(word)
            current_width += word_width + space
    
    # Render the last line of text
    if current_line:
        line_surface = font.render(' '.join(current_line), True, color)
        screen.blit(line_surface, (x + (max_width - line_surface.get_width()) // 2, y))


def render_multiline_left_justified_textbox(screen, texts, pos, font, color):
    x, y = pos
    y_spacing = 20  # Vertical spacing between lines
    for text in texts:
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (x, y))
        y += y_spacing

def main():
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))
    font = pygame.font.Font('resources/fonts/NeueHaasDisplayMediu.ttf', 26)  # Larger font for algorithm buttons and map name
    description_font = pygame.font.Font('resources/fonts/NeueHaasDisplayRoman.ttf', 14)  # Smaller font for descriptions
    map_txt = sys.argv[1]
    map_name = map_txt.split('/')[-1].split('.')[0]
    map_data = parse_map(map_txt)
    images = load_images(TILE_SIZE)
    initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'])

    current_state = initial_state
    algorithm_finished = False
    current_algorithm = ''

    while True:
        choice = main_menu(screen, font, description_font, map_name, current_state, images, map_data, algorithm_finished)
        current_algorithm = choice
        if choice in ['BFS', 'DFS', 'Global Greedy', 'Local Greedy', 'A*', 'IDDFS']:
            pygame.display.flip()

            # Algorithm execution
            initial_node = Node(current_state, None, None, 0)
            start_time = time.time()
            search_result, expanded_nodes, frontier_count  = run_algorithm(choice, initial_node)
            end_time = time.time() - start_time

            results_text = (
                f"{current_algorithm} took {end_time:.2f}s",
                f"Nodes Expanded: {expanded_nodes}",
                f"Frontier Size: {frontier_count}"
            )

            render_multiline_left_justified_textbox(screen, results_text, (50, 320), description_font, (255, 255, 255))

            for action in search_result:
                dx, dy = action
                new_state = current_state.move_player(dx, dy)
                if new_state:
                    current_state = new_state
                draw_board(screen, current_state, images, TILE_SIZE, 350, 150, map_data)
                pygame.display.flip()
                pygame.time.wait(100)
            algorithm_finished = True

        elif choice == 'Reset':
            current_state = initial_state
            algorithm_finished = False  # Reset the flag so algorithms can run again
            current_algorithm = ''

if __name__ == "__main__":
    main()
