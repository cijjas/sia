import pygame
import sys
import string

pygame.init()

# Grid size (you can change 'n' to any size you prefer)
n = 5  # For example, a 5x5 grid

# Cell dimensions
cell_size = 50
margin = 1

# Window dimensions
window_width = n * (cell_size + margin) + margin
window_height = (
    n * (cell_size + margin) + margin + 150
)  # Extra space for instructions and button
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Draw Letters")

# Font for instructions and button
font = pygame.font.Font(None, 24)

# List of uppercase English letters
letters = list(string.ascii_uppercase)

# Dictionary to store grids for each letter
letter_grids = {}

# Button dimensions and position
button_width, button_height = 100, 30
button_x = (window_width - button_width) // 2
button_y = window_height - 50


def draw_grid(grid):
    """Draw the grid on the screen."""
    for row in range(n):
        for col in range(n):
            color = (255, 255, 255) if not grid[row][col] else (0, 0, 0)
            pygame.draw.rect(
                screen,
                color,
                [
                    (margin + cell_size) * col + margin,
                    (margin + cell_size) * row + margin,
                    cell_size,
                    cell_size,
                ],
            )


def draw_button():
    """Draw the reset button on the screen."""
    pygame.draw.rect(
        screen, (180, 180, 180), (button_x, button_y, button_width, button_height)
    )
    button_text = font.render("Reset", True, (0, 0, 0))
    screen.blit(button_text, (button_x + 20, button_y + 5))


# Main loop for each letter
for letter in letters:
    # Initialize the grid for the current letter
    grid = [[False for _ in range(n)] for _ in range(n)]
    drawing = True
    while drawing:
        mouse_pressed = pygame.mouse.get_pressed()[
            0
        ]  # Check if the left mouse button is pressed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Move to the next letter
                    drawing = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the reset button was clicked
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if (
                    button_x <= mouse_x <= button_x + button_width
                    and button_y <= mouse_y <= button_y + button_height
                ):
                    # Clear the grid for the current letter
                    grid = [[False for _ in range(n)] for _ in range(n)]

        # Get the mouse position and update the grid if the mouse is pressed
        if mouse_pressed:
            x, y = pygame.mouse.get_pos()
            col = x // (cell_size + margin)
            row = y // (cell_size + margin)
            if 0 <= row < n and 0 <= col < n:
                grid[row][col] = True  # Set the cell as drawn

        # Clear the screen
        screen.fill((200, 200, 200))

        # Render instructions in separate lines
        instruction1 = font.render(f"Draw letter: {letter}", True, (0, 0, 0))
        instruction2 = font.render("Click and hold to draw", True, (0, 0, 0))
        instruction3 = font.render("Press ENTER when done", True, (0, 0, 0))

        # Position the instructions at the bottom
        screen.blit(instruction1, (10, n * (cell_size + margin) + margin))
        screen.blit(instruction2, (10, n * (cell_size + margin) + margin + 30))
        screen.blit(instruction3, (10, n * (cell_size + margin) + margin + 60))

        # Draw the grid
        draw_grid(grid)

        # Draw the reset button
        draw_button()

        # Update the display
        pygame.display.flip()

    # Store the grid for the current letter
    letter_grids[letter] = [row[:] for row in grid]  # Deep copy

from datetime import datetime

# Exit pygame
pygame.quit()

# Ask for filename to save the bitmaps
filename = input("Enter filename to save (leave blank for default): ")

# If no filename is given, set a default name with the current date
if not filename.strip():
    current_date = datetime.now().strftime("%Y%m%d")
    filename = f"letters_{current_date}.txt"

with open(filename, "w") as f:
    for letter in letters:
        f.write(f"= {letter}\n")
        grid = letter_grids[letter]
        for row in grid:
            line = "".join(["*" if cell else " " for cell in row])
            f.write(line + "\n")

print(f"Bitmaps saved to {filename}")
