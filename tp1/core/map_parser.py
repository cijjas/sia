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

def find_corners(map_data):
    corners = []
    for space in map_data['spaces']:
        x, y = space
        if (x + 1, y) in map_data['walls'] and (x, y + 1) in map_data['walls']:
            corners.append(space)
        elif (x - 1, y) in map_data['walls'] and (x, y + 1) in map_data['walls']:
            corners.append(space)
        elif (x + 1, y) in map_data['walls'] and (x, y - 1) in map_data['walls']:
            corners.append(space)
        elif (x - 1, y) in map_data['walls'] and (x, y - 1) in map_data['walls']:
            corners.append(space)
    return corners

def parse_map(map_file):
    with open(f'{map_file}', 'r') as f:
        lines = f.readlines()

    map_data = {
        'height' : len(lines),
        'width' : max(len(line) for line in lines) - 1,
        'walls' : [],
        'goals' : [],
        'boxes' : [],
        'corners' : [],
        'spaces' : [],
        'player' : None,
    }

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == map_object['wall']:
                map_data['walls'].append((x, y))
            elif char == map_object['goal']:
                map_data['goals'].append((x, y))
            elif char == map_object['box']:
                map_data['boxes'].append((x, y))
            elif char == map_object['player']:
                map_data['player'] = (x, y)
            elif char == map_object['space']:
                map_data['spaces'].append((x, y))
    
    map_data['corners'] = find_corners(map_data)

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
    
