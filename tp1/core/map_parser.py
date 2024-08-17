import sys

map_object = {
    "player": "@",
    "wall": "#",
    "goal": ".",
    "box": "$",
    "player_on_goal": "p",
    "box_on_goal": "*",
    "space": " ",
}

def parse_map(map_file):
    with open(f'{map_file}', 'r') as f:
        lines = f.readlines()

    map_data = {
        'height' : len(lines),
        'width' : max(len(line) for line in lines) - 1,
        'walls' : set(),
        'goals' : set(),
        'boxes' : set(),
        'spaces' : set(),
        'player' : None,
    }

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == map_object['wall']:
                map_data['walls'].add((x, y))
            elif char == map_object['goal']:
                map_data['goals'].add((x, y))
            elif char == map_object['box']:
                map_data['boxes'].add((x, y))
            elif char == map_object['player']:
                map_data['player'] = (x, y)
            elif char == map_object['space']:
                map_data['spaces'].add((x, y))
            elif char == map_object['player_on_goal']:
                map_data['player'] = (x, y)
                map_data['goals'].add((x, y))
            elif char == map_object['box_on_goal']:
                map_data['boxes'].add((x, y))
                map_data['goals'].add((x, y))

    return map_data

def print_map(map_data):
    for y in range(map_data['height']):
        for x in range(map_data['width']):
            if (x, y) in map_data['walls']:
                print('#', end='')
            elif (x, y) in map_data['goals']:
                print('.', end='')
            elif (x, y) in map_data['boxes']:
                print('$', end='')
            elif (x, y) == map_data['player']:
                print('@', end='')
            else:
                print(' ', end='')
        print()
