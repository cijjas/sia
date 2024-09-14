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

- __`game_config.json`__ contiene la configuracion del juego, con los objetos de
   - `character_classes`: Un array que contiene las clases del juego (strings) que se pueden seleccionar
   - `points_range`: Un array de dos elementos integer que indican el rango de valores de puntos que se pueden obtener
   - `time_limit_range`: Un array de dos elementos integer que indican el rango de tiempo que se pueden otorgar para asignar los puntos (en segundos)

- __`algorithm_config.json`__ contiene la configuracion de los algoritmos geneticos, contiene los objetos
   - `population_size` un integer que indica el tamaño de la poblacion
   - `operators` un objeto que contiene los metodos de cruza y mutacion
      - `crossover` un objeto con informacion del metodo de cruza
         - `method` el metodo de cruza a utilizar. Las opciones son `single_point`, `two_point`, `uniform`, `annular`.

      - `mutation` un objeto con informacion del metodo de mutacion
         - `method` el metodo de mutacion a utilizar. Las opciones son `gen`, `multigen`, `multigen_limited`, `complete`.
         - `distribution` la distribucion que se usara para modificar las stats. Las opciones son `uniform`, `gaussian`, `beta`, `gamma`.
         - `distribution_params` los parametros que recibe la distribucion indicada, varian segun la `distribution` elegida. Los campos del objeto varian segun la funcion elegida en distribution
            - `gaussian(std_p, std_h)`: donde la desviación estandar corresponde a puntos o altura respectivamente. (los puntos operan en otro dominio que la altura)
            - `beta(alpha, beta)`: favorece el crecimiento
            - `gamma(shape)`: favorece el decrecimiento y la escala se define automáticamente por el valor que se está modificando

         - `rate` objeto que contiene la informacion de tasa de mutacion (siempre ligada a la generación)
            - `method` El metodo para calcular la tasa de mutacion dinamica entre generaciones. Las opciones son `constant`, `sinusoidal`, `exponential_decay`
            - `initial_rate` La tasa de mutacion en la generacion cero
            - `final_rate` El minimo que puede alcanzar la tasa de mutacion, campo ignorado para method `constant`
            - `decay_rate` Una constante que se utiliza en el calculo de la tasa de mutacion. Campo únicamente considerado para `exponential_decay`
            - `period` **Campo ignorado salvo para method `sinusoidal`**. Indica el periodo de la funcion sinusoidal.

   - `selection` un objeto que contiene informacion de seleccion de individuos en una generacion
      - `selection_rate` la tasa de individuos a seleccionar
      - `parents` un array de objetos que contienen la informacion de todos los metodos de seleccion que se utilizaran para la generacion
         - `method` el metodo de seleccion de padres. Las opciones son `elite`, `roulette`, `ranking`, `universal`, `deterministic_tournament`, `probabilistic_tournament`, `boltzmann`
         - `weight` El peso normalizado de cada metodo de seleccion. La suma de todos los weights en elementos del array parents debe resultar 1
         - `params` objeto opcional que indica parametros adicionales del metodo, de ser necesarios
            - Torneo deterministico: `tournament_size`  Indica la cantidad de “batallas”
            - Torneo probabilístico: `threshold`: 
            - Boltzmann: `t_0`, `t_C`, `k`

      - `replacement` : idema `parents` pero con otro objetivo
         - `method` el metodo de seleccion de reemplazo. Las opciones son `elite`, `roulette`, `ranking`, `universal`, `deterministic_tournament`, `probabilistic_tournament`, `boltzmann`
         - `weight` El peso normalizado de cada metodo de seleccion. La suma de todos los weights en elementos del array replacemente debe resultar 1

   - `termination_criteria` objeto que contiene informacion sobre las diferentes condiciones de corte del algoritmo
      - `max_generations` cantidad de generaciones maximas a generar

- **`placeholder.json`** es un archivo opcional que indica los atributos que deberia tener la poblacion inicial
   - `ignore` Valor booleano que identifica si los contenidos del archivo deberian ser ignorados en la generacion de poblacion inicial
   - `strength`, `dexterity`, `intelligence`, `vigor`, `constitution` Valores entre 0 y 1 cuya sumatoria debe ser igual a 1. Indican la proporcion de puntos que se asignaran a la estadistica que comparte su nombre.
   - `height` Objeto que contiene en rango de alturas otorgadas a los individuos de la poblacion inicial
      - `min`, `max` el piso y el techo, respectivamente, de altura. `min` debe ser menor o igual a `max`, con un minimo de 1.3. `max` debe ser como maximo 2.

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