# Tabla de Contenidos
1. [Ejercicio I - 8-puzzle](#ejercicio-i---8-puzzle)
   - [8-Puzzle](#8-puzzle)
   - [8-Puzzle data generation](#8-puzzle_data_generation)
2. [Ejercicio II - Sokoban](#ejercicio-ii---sokoban)
   - [Intro](#intro)
   - [Sokoban cool](#sokoban-cool)
   - [Sokoban data generation](#sokoban-data-generation)
   - [Sokoban data analysis](#sokoban-data-analysis)
3. [Q&A sobre la implementación](#qa-sobre-la-implementación)
   - [¿Cómo implementamos backtracking?](#cómo-implementamos-backtracking)
   - [¿Qué hacemos con repetidos estados?](#qué-hacemos-con-repetidos-estados)
4. [Trabajos futuros y referencias](#trabajos-futuros-y-referencias)


# Ejercicio I - 8-puzzle
Para instalar las dependencias usar pipenv

```sh
pipenv install
```

## 8-Puzzle
Todo el código de la implementación del 8-Puzzle se encuentra en [8-puzzle](8-puzzle.py). Los archivos csv generados se guardan en [output](output/). El juego considera un estado inicial y estado final fijados en el código.

Para ejecutar el 8-Puzzle correr en el directorio tp1 el siguiente comando:

```sh
python3 8-puzzle.py
```

## 8-Puzzle_data_generation

Este ejecutable genera los gráficos en la carpeta [output/graphs](output/graphs) en base a los archivos csv generados por juego.

Para correrlo, ejecutar en el directorio tp1 el siguiente comando:
```sh
python3 8-puzzle_data_generation.py
```

# Ejercicio II - Sokoban

## Intro

Todos los códigos para el funcionamiento escencial de las estructuras y algorítmos se encuentran en la carpeta core.

El tp1 contiene el código necesario para poder probar distintos algorítmos de búsqueda (algunos desinformados como otros informados). En este repo se encuentran BFS, DFS, A*, Local Greedy y Global Greedy, ver: [directorio de algorítmos](core/algorithms/). Para los informados (que requieren de una heurística) se pueden usar las heurísticas definidas en el [archivo de heuristicas](core/heuristics.py). Dentro de la carpeta [models](core/models/) se encuentra la definición de estado y nodo.


La carpeta maps permite definir mapas como se definen en [este página](http://www.game-sokoban.com/index.php?mode=level&lid=200) que son parseados por el [map parser](core/utils/map_parser.py).

Hay varias formas de ejecución para el sokoban.

1. Modo interactivo [SOKBAN COOL](sokoban_cool.py)
2. Modo de generación de data [GENERATION](sokoban_data_generation.py)
3. Modo análsis [ANALYSIS](sokoban_data_analysis.py)

### Sokoban cool

Interfaz gráfica simple, hecha con pygame para probar distintos algorítmos y ver sus soluciones propuestas.

![bfs 1 gif](resources/gifs/bfs_1.gif)

para ejecutarlo

```sh
python3 sokoban_cool.py path/to/map.txt
```

Dado que el objetivo de este tp no era tener una interfaz super perfecta (sino que obtener soluciones de mapas de sokoban) dejamos algunas configuraciones a hacer manualmente como por ejemplo elegir las heurísticas a usar para los algoritmos informados. Por default estan M1 y Deadlocks. Para cambiar esto se puede configurar en la función de `run_algorithm` de [sokoban_cool.py](sokoban_cool.py) donde se pueden quitar o introducir nuevas heurísticas definidas en el archivo de heurísticas. También es posible cambiar el texture pack agregando el parámetro en la función `load_images` mencionando el nombre de la carpeta que se quiera que está en [texture_packs](resources/texture_packs/).

### Sokoban data generation

Para usar data generation se debe definir un archivo de configuración con especificaciones de los métodos de corrida como el mapa, los algoritmos, las heurísticas y las iteraciones. También se define un archivo de salida en formato csv que es donde se vuelcan los resultados de las ejecuciones con información como el tiempo de ejecución, cantidad de nodos expandidos, solución, costo de la solución entre otros. En [configs](configs/) se encuentran varios de los usados para el análisis y la presentación.


Por ejemplo:
```json
{
    "executions": [
        {
            "output_file": "name_of_the_output_file.csv",
            "data_generation": {
                "maps": [ "map1.txt", "map2.txt"]
                "algorithms": [
                    {
                        "name": "ALGORITHM_1",
                        "heuristics": ["Heuristic_1", "Heuristic_2"]
                    },
                    {
                        "name": "ALGORITHM_2",
                        "heuristics": ["Heuristic_1"]
                    }
                ],
                "iterations_for_average": 10
            }
        }
    ]
}

```
donde todas las configuraciones deben ser de esta forma.

|campo | opciones|
|--|--|
| output_file| cualquier nombre seguido de .csv|
| maps | array de nombres de mapas a analizar (se buscan en maps/ asi que definirlos ahí)|
| algorithms | array de algoritmos |
| name | BFS, DFS, A_STAR, GREEDY_LOCAL, GREEDY_GLOBAL |
| heuristics | M1, M2, M3, Smarthattan, Deadlocks, Basic, Euclidean, Inadmissible|
| iterations_for_average | cantidad de ejecuciones de los algorítmos \*|

\* notar que esto sólo debería cambiar el tiempo de ejecución dado que es el único parámetro no determinístico para una ejecución


Correr de la siguiente manera:

```sh
python3 sokoban_data_generation.py configs/my_configuration.json
```

El output csv de esta ejecución se guarda en la carpeta [output](output/) que está configurada para no commitearse.

### Sokoban data analysis

Esta ejecución nos permite hacer el análisis del paso previo (data generation) para generar gráficos comparativos de las distintas ejecuciones. Su propósito es ser configurado y perfeccionado a medida del que hace el análisis.

## Q&A sobre la implementación

### ¿Cómo implementamos backtracking?
Depende del algoritmo a usar pero sepuede ver en las implementaciones en el directorio de algoritmos que se usan setructuras tipo colas y stacks (algunos ordenados) como para mantener nodos con sus estados a la espera de ser popeados y explorados. De esta forma garantizando backtracking.


### ¿Qué hacemos con repetidos estados?

Nos aseguramos de no visitar nodos repetidos manteniendo un set paralelo a la ejecutción que garantiza evitar la repetición y evita entrar en loops infinitos.


# Trabajos futuros y referenicas

- [Experto en Heuristicas](https://www.reddit.com/r/algorithms/comments/fedzu1/pathfinding_heuristic_for_indirect_movement_like/)
- IDA*
- IDDFS
- Preprocesamiento de heurísticas para lograr O(1)
