import pygame
from core.state import State
from core.map_parser import parse_map
from core.search_engine import search
from core.algorithms import DFS, BFS

import sys


TILE_SIZE = 64



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
    screen.blit(images['@'], (player[0] * tile_size, player[1] * tile_size))



def main():
    pygame.init()
    map_data = parse_map(sys.argv[1])
    map_width = map_data['width']
    map_height = map_data['height']

    game_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'])
    screen = pygame.display.set_mode((map_width* TILE_SIZE, map_height * TILE_SIZE))
    
    
    images = load_images(TILE_SIZE)
    clock = pygame.time.Clock()
    running = True

    def inner_draw_board(state:State):
        draw_board(screen, state, images, TILE_SIZE, map_width, map_height)
        pygame.display.flip()
    
    search_result = search(BFS(), game_state, inner_draw_board)

    draw_board(screen, search_result, images, TILE_SIZE, map_width, map_height)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                break

#    while running:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                running = False
#            elif event.type == pygame.KEYDOWN:
#                if event.key == pygame.K_DOWN:
#                    new_state = game_state.move_player(0, 1)
#                    if new_state:
#                        game_state = new_state
#                elif event.key == pygame.K_RIGHT:
#                    new_state = game_state.move_player(1, 0)
#                    if new_state:
#                        game_state = new_state
#                elif event.key == pygame.K_UP:
#                    new_state = game_state.move_player(0, -1)
#                    if new_state:
#                        game_state = new_state
#                elif event.key == pygame.K_LEFT:
#                    new_state = game_state.move_player(-1, 0)
#                    if new_state:
#                        game_state = new_state
#                
#        draw_board(screen, game_state, images, TILE_SIZE, map_width, map_height)
#        pygame.display.flip()
#        clock.tick(60)
    
    pygame.quit()
    

if __name__ == "__main__":
    main()
