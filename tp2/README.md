# Introducción
Motor de algoritmos genéticos para el análisis de escenarios y resolución del juego de rol de fantasía medieval "ITBUM ONLINE".

# Instrucciones de ejecución
Ejecutar

```sh
python main.py config_file_path.json
```
- **`game_config.json`** contiene la configuracion del juego, con los objetos de
    - `character_classes`: Un array que contiene las clases del juego (strings) que se pueden seleccionar
    - `points_range`: Un array de dos elementos integer que indican el rango de valores de puntos que se pueden obtener
    - `time_limit_range`: Un array de dos elementos integer que indican el rango de tiempo que se pueden otorgar para asignar los puntos (en segundos)
- **`algorithm_config.json`** contiene la configuracion de los algoritmos geneticos, contiene los objetos
    - `population_size` un integer que indica el tamaño de la poblacion
    - `operators` un objeto que contiene los metodos de cruza y mutacion
        - `crossover` un objeto con informacion del metodo de cruza
            - `method` el metodo de cruza a utilizar. Las opciones son `single_point`, `two_point`, `uniform`, `annular`.
        - `mutation` un objeto con informacion del metodo de mutacion
            - `method` el metodo de mutacion a utilizar. Las opciones son `gen`, `multigen`, `multigen_limited`, `complete`.
            - `distribution` la distribucion que se usara para modificar las stats. Las opciones son `uniform`, `gaussian`, `exponential`, `beta`, `gamma`.
            - `distribution_params` los parametros que recibe la distribucion indicada, varian segun la `distribution` elegida. Los campos del objeto varian segun la funcion elegida en distribution
                - `gaussian(mean, std)`
                - `exponential(lambda)`
                - `beta(alpha, beta)`
                - `gamma(shape, scale)`
            - `rate` objeto que contiene la informacion de tasa de mutacion
                - `method` El metodo para calcular la tasa de mutacion dinamica entre generaciones. Las opciones son `constant`, `sinusoidal`, `exponential_decay`
                - `initial_rate` La tasa de mutacion en la generacion cero
                - `final_rate` El minimo que puede alcanzar la tasa de mutacion, campo ignorado para method `constant`
                - `decay_rate` Una constante que se utiliza en el calculo de la tasa de mutacion. Campo ignorado para method `constant`
                - `period` **Campo ignorado salvo para method `sinusoidal`**. Indica el periodo de la funcion sinusoidal.
    - `selection` un objeto que contiene informacion de seleccion de individuos en una generacion
        - `selection_rate` la tasa de individuos a seleccionar
        - `parents` un array de objetos que contienen la informacion de todos los metodos de seleccion que se utilizaran para la generacion
            - `method` el metodo de seleccion de padres. Las opciones son `elite`, `roulette`, `ranking`, `universal`, `deterministic_tournament`, `probabilistic_tournament`, `boltzmann`
            - `weight` El peso normalizado de cada metodo de seleccion. La suma de todos los weights en elementos del array parents debe resultar 1
            - `params` objeto opcional que indica parametros adicionales del metodo, de ser necesarios
                - `tournament_size` **Parametro del metodo de torneo deterministico.** Indica la cantidad de “batallas”
        - `replacement`
            - `method` el metodo de seleccion de reemplazo. Las opciones son `elite`, `roulette`, `ranking`, `universal`, `deterministic_tournament`, `probabilistic_tournament`, `boltzmann`
            - `weight` El peso normalizado de cada metodo de seleccion. La suma de todos los weights en elementos del array replacemente debe resultar 1
    - `termination_criteria` objeto que contiene informacion sobre las diferentes condiciones de corte del algoritmo
        - `max_generations` cantidad de generaciones maximas a generar
- **`placeholder.json`** es un archivo opcional que indica los atributos que deberia tener la poblacion inicial
  - `ignore` Valor booleano que identifica si los contenidos del archivo deberian ser ignorados en la generacion de poblacion inicial
  - `strength`, `dexterity`, `intelligence`, `vigor`, `constitution` Valores entre 0 y 1 cuya sumatoria debe ser igual a 1. Indican la proporcion de puntos que se asignaran a la estadistica que comparte su nombre.
  - `height` Objeto que contiene en rango de alturas otorgadas a los individuos de la poblacion inicial
    - `min`, `max` el piso y el techo, respectivamente, de altura. `min` debe ser menor o igual a `max`, con un minimo de 1.3. `max` debe ser como maximo 2.