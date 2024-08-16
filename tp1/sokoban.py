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

def draw_text(screen, text, font, color, position):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def read_board_from_file(file_path):
    with open(file_path, 'r') as file:
        return [list(line.strip()) for line in file.readlines()]

def main():
    pygame.init()
    tile_size = 36
    board_file_path = 'test/test_5'
    initial_board = read_board_from_file(board_file_path)
    
    num_rows = len(initial_board)
    num_cols = max(len(row) for row in initial_board)
    width, height = num_cols * tile_size, num_rows * tile_size
    
    screen = pygame.display.set_mode((width, height))
    images = load_images(tile_size)

    font = pygame.font.Font(None, 36)  # None uses the default font, 36 is the font size
    
    running = True
    clock = pygame.time.Clock()

    def reset_game():
        return State([list(row) for row in initial_board])

    game_state = reset_game()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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
                elif event.key == pygame.K_r:
                    game_state = reset_game()
                elif event.key == pygame.K_p:
                    game_state.undo_box_action()
            if game_state.is_solved():
                print('You won!')
                running = False

        draw_board(screen, game_state, images, tile_size)
        draw_text(screen, "Press 'p' to undo, 'r' to restart", font, (255, 255, 255), (10, height - 40))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
