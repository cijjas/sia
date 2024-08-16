import pygame
from core.state import State
from algorithms.bfs import bfs_solve
from algorithms.dfs import dfs_solve
import json
import sys

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


def display_win_message(screen):
    font = pygame.font.Font(None, 32)  
    text = font.render('Ganaste! Press R to Restart', True, (255, 255, 0))  # Yellow color
    text_rect = text.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))
    screen.blit(text, text_rect)
    pygame.display.flip() 


def load_map_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['map']

def main():
    pygame.init()
    
    map_data = load_map_from_json(sys.argv[1]) 
    game_state = State(map_data)


    tile_size = 64
    width, height = len(game_state.board[0]) * tile_size, len(game_state.board) * tile_size
    screen = pygame.display.set_mode((width, height))
    images = load_images(tile_size)
    
    solution_path = bfs_solve(game_state)
    if solution_path is None:
        print("No solution found.")
        pygame.quit()
        return
    
    print("Solution path:", solution_path)
    
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
                    draw_board(screen, game_state, images, tile_size)
                    pygame.display.flip()
        
        if path_index < len(solution_path):
            direction = solution_path[path_index]
            key_event = pygame.event.Event(pygame.KEYDOWN, key=direction_to_key[direction])
            pygame.event.post(key_event)
            path_index += 1
            pygame.time.delay(500)  # Add delay to see each move

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
