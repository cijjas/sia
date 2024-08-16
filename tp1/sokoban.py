import pygame
from core.state import State

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


# TODO tambien hay que ver el tema que cuando pasa por arriba dle goal lo borra me parece
def main():
    pygame.init()
    tile_size = 64
    width, height = 15 * tile_size, 10 * tile_size
    screen = pygame.display.set_mode((width, height))
    images = load_images(tile_size)

    # Game loop setup
    running = True
    clock = pygame.time.Clock()
    board = [
        "      ###      ",
        "      #.#      ",
        "  #####.#####  ",
        " ##         ## ",
        "##  # # # #  ##",
        "#  ##     ##  #",
        "# ##  # #  ## #",
        "#     $@$     #",
        "####  ###  ####",
        "   #### ####   ",
    ]
    game_state = State([list(row) for row in board])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game_state.move_player(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    game_state.move_player(1, 0)
                elif event.key == pygame.K_UP:
                    game_state.move_player(0, -1)
                elif event.key == pygame.K_DOWN:
                    game_state.move_player(0, 1)

        draw_board(screen, game_state, images, tile_size)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
