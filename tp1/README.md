# Ejercicio I - 8-Puzzle

### ¿Que estructura de datos utilizarian?
Matriz de 3x3 donde cada valor es un char y el vacio es representado por el '\0'. Es decir, cada número del tablero lo representamos como char (1 es '1', 2 es '2', y así)
La igualdad de estados se realiza con un previo ordenamiento, ubicando en la esquina superior izquierda el valor minimo entre las 4 posibles esquinas.
Por ejemplo
3 2 1
4 0 6
7 8 5
Se reordena a la matriz a la forma canonica definida
1 6 5
2 0 8
3 4 7
De esta manera, podemos comparar rotaciones del tablero que pueden llegar a representar el mismo estado.

### Heuristicas Admisibles
- Sumatoria de distancias manhattan, aunque seria valido para norma = 0 (euclideana), norma supremo
- Sumatoria del triangulo superior sea menor
- Cantidad de posicionamientos correctos de los numeros, hacer una desviacion estandar segun la posicion de donde estas y donde deberia estar
- Cantidad de posicionamientos incorrectos de los numeros
- v siendo la lista de elementos
$$\sum_{i = 1}^{8} \frac{v[i]}{8\cdot i }-1$$

### ¿Que Algoritmo de Busqueda usarian y con que heuristicas?
Cuando hagamos el Sokoban vamos a tener mas idea

# Ejercicio II - Sokoban
### Definicion de Estados
Matriz de NxM de posicion donde ademas se incluye el elemento en esa posicion, el elemento puede ser:
- Jugador
- Pared
- Libre
- Caja
- Posicion objetivo de la caja
- Caja en lugar correcto
Se le suman estados logicos utiles para optimizar el BFS
- Estados repetidos para no hacer loops infinitos
- Estados bloqueados para que el BFS los descarte instantaneamente
### Key Points:
- Usar arcade library para generar los mapas
- Definir un buen equals de estados que optimize la busqueda, constantemente va a estar revisando si alguno de los posibles estados es repetido
### Heuristicas
- Comparar siempre con la respuesta optima del BFS, si no es la misma entonces es un heuristica inadmisible
- Toda heuristica genera un trade off entre memoria y tiempo, procesas mas profundamente pero explotas menos nodos
- Hay heuristicas inadmisibles que son muy fuertes en ciertos mapas, son interesantes para plantear




# ISSUES
- Expanded nodes del DFS es mayor al del BFS, imposible que ocurra practicamente (correr con bug_config.json)


- backtracking como lo implementamos
- que hacemos con repetidos
- detalles de implementación izq der
- dfs importante como decidis el siguiente estado (no trivial) hay algunos tableros que dependen de esta decisión.
- checkeo de deadlocks
- a* debería dar óptimo para todas las heurisiticas admisibles

[a star level 2](resources/gifs/a_star_2.gif)

# REF

- [Experto en Heuristicas](https://www.reddit.com/r/algorithms/comments/fedzu1/pathfinding_heuristic_for_indirect_movement_like/)
- IDA*
- Poner cantidad de iteraciones en los graficos, y toda la informacion fija
- Usar otro color scheme para los graficos
- ponerte a un escalon de la solucion y que de mas la heuristica no admisible
- estado definido como posicion de las cajas y jugador
- Creas Mapa entonces tenes Walls, Targets
