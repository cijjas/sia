# Introducción
Motor de algoritmos genéticos para el análisis de escenarios y resolución del juego de rol de fantasía medieval "ITBUM ONLINE".

# Instrucciones de ejecución
Ejecutar

```sh
python main.py config_file_path.json
```

Ejemplo de archivo de configuración:


| Sección       | Clave            | Descripción                                          | Posibles Valores                                                          | Parámetros Adicionales                        |
| ------------- | ---------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------- |
| **Operators** |                  |                                                      |                                                                           |                                               |
|               | crossover.method | Método utilizado para el cruce entre individuos.     | `single_point`, `two_point`, `uniform`, `annular`                         |                                               |
|               | mutation.method  | Método utilizado para la mutación de los individuos. | `gen_uniform`, `multigen_uniform`, `multigen_uniform_limited`, `complete` | `multigen_uniform_limited`: `amount` (entero) |
|               | mutation.rate    | Tasa de mutación aplicada durante la simulación.     | `[0-1]`                                                                   |                                               |

| **Selection**            | selection_rate        | Porcentaje de la población seleccionada para reproducción.                                         | `[0-1]`                                                                                                    |                                                                                                                           |
| ------------------------ | --------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
|                          | parents.method        | Array de métodos para la selección de padres.                                                      | `elite` `roulette` `deterministic_tournament` `probabilistic_tournament` `ranking` `universal` `boltzmann` | `deterministic_tournament`: `tournament_size` (int) `probabilistic_tournament`: `threshold` `boltzmann`: `t_0` `t_C` `k`  |
|                          | parents.method.weight | Porción de los padres seleccionados con dicho método (la suma de las porciones siempre debe dar 1) | `[0-1]`                                                                                                    |                                                                                                                           |
|                          |                       |                                                                                                    |                                                                                                            |                                                                                                                           |
|                          | replacement.method    | Métodos para la reposición de la población.                                                        | Idem parentrs.method                                                                                       |                                                                                                                           |
| **Termination Criteria** | max_generations       | Número máximo de generaciones antes de terminar.                                                   | 3                                                                                                          |                                                                                                                           |
|                          | max_time              | Tiempo máximo (en segundos) antes de terminar.                                                     | 90                                                                                                         |                                                                                                                           |
|                          | structure             | Estructura de la condición de terminación.                                                         | portion, generations                                                                                       |                                                                                                                           |
|                          | content               | Generaciones sin cambio significativo.                                                             | 10                                                                                                         |                                                                                                                           |
|                          | desired_fitness       | Aptitud deseada para la terminación.                                                               | 0.9                                                                                                        |                                                                                                                           |
| **Fixed Population**     | -                     | Población inicial fija con atributos específicos.                                                  | Lista de atributos (fuerza, destreza, etc.)                                                                |                                                                                                                           |

