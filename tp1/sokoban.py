import pygame
from core.state import State
import core.algorithms as alg

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic



def load_images(tile_size):
    images = {
        '#': pygame.image.load('core/resources/wall.jpg').convert_alpha(),
        '@': pygame.image.load('core/resources/player.jpg').convert_alpha(),
        '$': pygame.image.load('core/resources/box.png').convert_alpha(),
        '.': pygame.image.load('core/resources/goal.png').convert_alpha(),
        '*': pygame.image.load('core/resources/box_on_goal.png').convert_alpha() # hay que ver que esto todavia no funca
    }

    for key in images:
        images[key] = pygame.transform.scale(images[key], (tile_size, tile_size))
    return images

def draw_board(screen, game_state, images, tile_size):
    screen.fill((0, 0, 0)) 
    for y, row in enumerate(game_state.board):
        for x, char in enumerate(row):
            if char in images:
                screen.blit(images[char], (x * tile_size, y * tile_size))

def draw_menu(screen, font, menu_options, selected_option):
    screen.fill((0, 0, 0))
    for i, option in enumerate(menu_options):
        color = (255, 255, 255) if i == selected_option else (100, 100, 100)
        draw_text(screen, option, font, color, (100, 100 + i * 40))
    pygame.display.flip()

def draw_text(screen, text, font, color, position):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def read_board_from_file(file_path):
    with open(file_path, 'r') as file:
        return [list(line.strip()) for line in file.readlines()]

def main():
    pygame.init()
    tile_size = 36
    board_file_path = 'test/test_4'
    initial_board = read_board_from_file(board_file_path)
    solve_functions = [alg.bfs, alg.dfs]
    menu_active = False
    menu_options = ["BFS", "DFS"]
    selected_option = 0

    min_width = 8
    min_height = 10
    
    num_rows = max(len(initial_board), min_height)
    num_cols = max(max(len(row) for row in initial_board), min_width)
    width, height = num_cols * tile_size, num_rows * tile_size
    
    screen = pygame.display.set_mode((width, height))
    images = load_images(tile_size)

    font = pygame.font.Font(None, 36)  # None uses the default font, 36 is the font size
        
    running = True
    clock = pygame.time.Clock()

    def reset_game():
        return State([list(row) for row in initial_board])

    game_state = reset_game()

    menu_active = False
    menu_options = ["BFS", "DFS"]
    selected_option = 0

    # Array of functions corresponding to the menu options
    solve_functions = [alg.bfs, alg.dfs]

    # Initialize solving and other variables
    solving = False
    solution_steps = []
    step_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    menu_active = not menu_active
                elif menu_active:
                    if event.key == pygame.K_UP:
                        selected_option = (selected_option - 1) % len(menu_options)
                    elif event.key == pygame.K_DOWN:
                        selected_option = (selected_option + 1) % len(menu_options)
                    elif event.key == pygame.K_RETURN:
                        print(f"Selected option: {menu_options[selected_option]}")
                        solution_steps = solve_functions[selected_option](game_state)
                        step_index = 0
                        solving = True
                        menu_active = False
                else:
                    if event.key == pygame.K_LEFT:
                        game_state.move_player(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        game_state.move_player(1, 0)
                    elif event.key == pygame.K_UP:
                        game_state.move_player(0, -1)
                    elif event.key == pygame.K_DOWN:
                        game_state.move_player(0, 1)
                    elif event.key == pygame.K_r:
                        game_state = reset_game()
                    elif event.key == pygame.K_p:
                        game_state.undo_box_action()
                    if game_state.is_solved():
                        print('You won!')
                        running = False

        if menu_active:
            draw_menu(screen, font, menu_options, selected_option)
        else:
            if solving and step_index < len(solution_steps):
                action = solution_steps[step_index]
                game_state.move_player(action[0], action[1])
                step_index += 1
                if step_index >= len(solution_steps):
                    solving = False
            draw_board(screen, game_state, images, tile_size)
            draw_text(screen, "Press 'p' to undo", font, (255, 255, 255), (10, height-30))
            draw_text(screen, "Press 'r' to reset", font, (255, 255, 255), (10, height-60))
            draw_text(screen, "Press 'e' to open menu", font, (255, 255, 255), (10, height-90))
            pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
