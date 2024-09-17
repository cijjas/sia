# Qué es esto?

Esto es un motor de algoritmos genéticos que tiene el objetivo de encontrar la mejor distribución de puntos para elegir un personaje en un juego de roles. En otras palabras tryhardear el juego y lograr ganarle al contrincante. Se simula una situación de juego en la cual se da $t: \text{tiempo}$ para distribuir $P: \text{puntos}$ para un personaje $C: \text{personaje}$. En el transcurso de $t$ se debe distribuir $P$ a lo largo de 5 atributos (`strenght`, `dexterity`, `intelligence`, `vigor`, `constitution`). Si se alcanza $t$ y no se eligió nada, se asigna aleatoriamente los puntos. 

El chiste está en que mientras transcurre $t$ nosotros ejecutamos un programa que tiene accesso a una función secreta `eve` que se usa para evalua la distribucón. De esta forma intentamos acercarnos al óptimo sobre una función que no concemos.

# Cómo ejecutarlo

Moverse a la carpeta `src` e instalar las dependencias

```sh {"id":"01J7RJCRBWSH7HRM3Y12KW5X34"}
cd src
pipenv install
```

Ejecutar el juego (que usa el motor)

```sh {"id":"01J7RJDT3K8CNAGYTA24KCQ3T3"}
pipenv run python3 master.py ../config/algorithm_config.json
```


# Hiperparámetros y configs

## `game_config.json`

Este archivo define el comportamiento del juego, por default solo se establecen parámentros fijos que responden al comportamiento originial del juego. Luego también hay una "seed" que se puede poner para que el juego siempre nos asigne el mismo valor de $t, P~ \text{y}~C$. 

| Clave              | Tipo        | Descripción                                                                                                   | Valores posibles                            |
|--------------------|-------------|---------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| `character_classes`| Array (String) | Un array que contiene las clases del juego que se pueden seleccionar                                          | Por default son warrior, archer, guardian, mage|
| `points_range`     | Array (Integer) | Un array de dos elementos que indican el rango de valores de puntos que se pueden obtener                     | `[0,200]` |
| `time_limit_range` | Array (Integer) | 10, 20 según los requerimeintos del juego                  | `[10, 120]`    |
|`seed.points`| int| puntos asignados por el juegoo para distribuir| 100 a 200|
|`seed.time`|int | define cuantos segundos te da el juego| 0 a 120|
|`seed.character`|string|define que caracter usas en el juego| `warrior`,`archer`, `guardian`, `mage` (cualquier opción de character_classes)|

>[!Note] 
> No tocar este archivo a menos que se quieran cambiar las reglas del juego. En todo caso sólo sacar o agregar el seed.

## `algorithm_config.json`

Este archivo contiene la configuración de los algoritmos genéticos.

### Parámetros Generales

| Clave             | Tipo   | Descripción                | Valores posibles / Opciones |
|-------------------|--------|----------------------------|-----------------------------|
| `population_size` | Entero | Tamaño de la población     | Cualquier entero positivo   |

### Operadores

#### Cruza

| Clave                        | Tipo   | Descripción                        | Valores posibles / Opciones                         |
|------------------------------|--------|------------------------------------|-----------------------------------------------------|
| `operators.crossover.method` | Cadena | Método de cruza a utilizar         | `single_point`, `two_point`, `uniform`, `annular`    |
|`operators.crossover.rate` | Flotante | Tasa de cambio si `uniform`, sino se ignora  | Valor entre 0 y 1 | 
#### Mutación

| Clave                                 | Tipo    | Descripción                                                            | Valores posibles / Opciones                         |
|---------------------------------------|---------|------------------------------------------------------------------------|-----------------------------------------------------|
| `operators.mutation.method`           | Cadena  | Método de mutación a utilizar                                          | `gen`, `multigen`, `multigen_limited`, `complete`   |
| `operators.mutation.distribution`     | Cadena  | Distribución para modificar las estadísticas                           | `uniform`, `gaussian`, `beta`, `gamma`              |
| `operators.mutation.distribution_params` | Objeto | Parámetros que recibe la distribución indicada                         | Ver sección de Parámetros de Distribución [Parámetros de Distribución](#parámetros-de-distribución)         |
| `operators.mutation.rate.method`      | Cadena  | Método para calcular la tasa de mutación dinámica entre generaciones   | `constant`, `sinusoidal`, `exponential_decay`       |
| `operators.mutation.rate.initial_rate` | Flotante | Tasa de mutación en la generación cero, también tasa de mutación para `constant`                               | Valor entre 0 y 1                                   |
| `operators.mutation.rate.final_rate`   | Flotante | Mínimo que puede alcanzar la tasa de mutación (ignorado si es `constant`) | Valor entre 0 y 1                                |
| `operators.mutation.rate.decay_rate`   | Flotante | Constante utilizada en el cálculo de la tasa de mutación (solo para `exponential_decay`) | Número positivo |
| `operators.mutation.rate.period`       | Flotante | Período de la función sinusoidal (solo para `sinusoidal`)             | Número positivo                                     |

##### Parámetros de Distribución

Los campos de `operators.mutation.distribution_params` varían según la distribución elegida:

- **`gaussian`**:
  - `std_p`: Desviación estándar para puntos.
  - `std_h`: Desviación estándar para altura.
- **`beta`**: Favorece el crecimiento
  - `alpha`: Parametro alpha (mayor corrimiento hacia la derecha de la función distribución, más crecimiento)
  - `beta`: Parámetro beta de la distribución.
- **`gamma`**: Favorece el decrecimiento
  - `shape`: (la escala se define automáticamente en base al valor siendo mutado).

### Selección y Reemplazo

| Clave                      | Tipo             | Descripción                         | Valores posibles / Opciones                                   |
|----------------------------|------------------|-------------------------------------|---------------------------------------------------------------|
| `selection.selection_rate` | Flotante         | Tasa de individuos a seleccionar    | Valor entre 0 y 1                                             |
| `selection.parents`        | Array de Objetos | Métodos de selección para padres    | Ver sección de [Métodos de Selección](#métodos-de-selección)                           |
| `selection.replacement`    | Array de Objetos | Métodos de selección para reemplazo | Ver sección de [Métodos de Selección](#métodos-de-selección)                           |

#### Métodos de Selección

Cada objeto dentro de `parents` o `replacement` incluye:

| Clave    | Tipo     | Descripción                                | Valores posibles / Opciones                                                        |
|----------|----------|--------------------------------------------|------------------------------------------------------------------------------------|
| `method` | Cadena   | Método de selección                        | `elite`, `roulette`, `ranking`, `universal`, `deterministic_tournament`, `probabilistic_tournament`, `boltzmann` |
| `weight` | Flotante | Peso que define que proporción de la selección o reemplazo va a ser considerada con este método\* | Valor entre 0 y 1                                                                  |
| `params` | Objeto   | Parámetros adicionales (opcional)          | Ver sección de [Parámetros del Método](#parámetros-del-método)                                               |

\* Si se tiene más de un método de selección o reemplazo la suma de estos `weight`s debe sumar 1 por ejemplo:

```json
"parents": [
            {
                "method": "boltzmann",
                "weight": 0.7,
                "params":
                {
                  "t_0": 0,
                  "t_C": 5,
                  "k":1
                }
            },
            {
                "method": "elite",
                "weight": 0.3
            }
        ],
```

##### Parámetros del Método

- **Torneo Determinístico (`deterministic_tournament`)**:
  - `tournament_size`: Número de "batallas".
- **Torneo Probabilístico (`probabilistic_tournament`)**:
  - `threshold`: Umbral para selección.
- **Boltzmann (`boltzmann`)**:
  - `t_0`: Temperatura inicial.
  - `t_C`: Temperatura de enfriamiento.
  - `k`: Constante de Boltzmann.

### Criterios de Terminación

| Clave                                  | Tipo   | Descripción                               | Valores posibles / Opciones |
|----------------------------------------|--------|-------------------------------------------|-----------------------------|
| `termination_criteria.max_generations` | Entero | Cantidad máxima de generaciones a generar | Cualquier entero positivo   |



## `initial_population_see.json`

| Clave                 | Tipo             | Descripción                                                                                             | Valores posibles                                      |
|-----------------------|------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `ignore`              | Boolean          | Indica si los contenidos del archivo deberían ser ignorados en la generación de población inicial        | `true`, `false`                                       |
| `strength`, `dexterity`, `intelligence`, `vigor`, `constitution` | Objetos | Contiene el rango de las estadísticas | `{"min: , "max":}`        |
| `min`, `max` (para `strength`, `dexterity`, etc.) | Float | Valores que representan el piso (`min`) y el techo (`max`) para la proporción de puntos                | `min` ≤ `max`, ambos entre 0 y 1                      |
| `height`              | Objeto           | Contiene el rango de alturas otorgadas a los individuos de la población inicial                          | `{ "min": mínimo de 1.3, "max": máximo de 2 }`        |
| `min`, `max` (para `height`) | Float    | El piso (`min`) y el techo (`max`) para la altura de los individuos. `min` debe ser menor o igual a `max` | `min` ≥ 1.3, `max` ≤ 2                                |
