# Aprendizaje No Supervisado: Redes Neuronales y Algoritmos de Reducción de Dimensionalidad

Este proyecto implementa varios modelos fundamentales de redes neuronales y algoritmos utilizados para aprendizaje no supervisado y reducción de dimensionalidad. Los modelos implementados incluyen:

- Red de Kohonen (Mapas Autoorganizados)
- Análisis de Componentes Principales (PCA)
- Regla de Oja
- Regla de Sanger
- Red de Hopfield

Cada algoritmo proporciona métodos únicos para el reconocimiento de patrones, agrupamiento y reducción de dimensionalidad, aplicables en múltiples áreas de aprendizaje automático e inteligencia artificial.

## Tabla de Contenidos

- [Instalación](#instalación)
- [Uso](#uso)
  - [Red de Kohonen](#red-de-kohonen)
  - [Análisis de Componentes Principales (PCA)](#análisis-de-componentes-principales-pca)
  - [Regla de Oja](#regla-de-oja)
  - [Regla de Sanger](#regla-de-sanger)
  - [Red de Hopfield](#red-de-hopfield)
- [Referencias](#referencias)

## Instalación

Para usar estos algoritmos, se requiere Python 3 y los siguientes paquetes como mínimo:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn` (opcional, para preprocesamiento de datos)

Instalación rápida:

```sh
pip install numpy pandas matplotlib scikit-learn
```

## Uso

Las implementaciones están disponibles en la carpeta [core](./src/core/). El conjunto de datos de Europa utilizado en los notebooks se encuentra en [./data/europe.csv](./data/europe.csv).

---

### Red de Kohonen

La red de Kohonen, o Mapas Autoorganizados (SOM), se emplea para el agrupamiento y la proyección de datos de alta dimensionalidad a una cuadrícula de menor dimensionalidad.

#### Hiperparámetros

```json
{
    "kohonen": {
        "seed": 42,
        "grid_size": 3,
        "learning_rate": 0.5,
        "eta_function": "exponential_decay",
        "initial_radius": 3,
        "radius_function": "exponential_decay",
        "number_of_iterations": 28000,
        "similarity_function": "euclidean"
    },
    "data": {
        "source": "../data/europe.csv",
        "scaling": "standard",
        "features": [
            "Area",
            "GDP",
            "Inflation",
            "Life.expect",
            "Military",
            "Pop.growth",
            "Unemployment"
        ]
    }
}
```

- `seed`: Valor de aleatoriedad para resultados replicables.
- `grid_size`: Tamaño de la cuadrícula que define la topología inicial de la red.
- `learning_rate`: Define la intensidad de ajuste en cada iteración.
- `eta_function`: Comportamiento del `learning_rate` a lo largo de las épocas (constante o `exponential_decay`).
- `initial_radius`: Radio inicial de vecindad.
- `radius_function`: Comportamiento del radio a lo largo de las épocas (constante o `exponential_decay`).
- `similarity_function`: Modo de calcular cercanía entre nodos (distancia euclídea, coseno o exponencial).
- `number_of_iterations`: Número de iteraciones para el ajuste de la red. Cada iteración representa el uso de un dato de entrada para modificar la red; una época implica el ajuste con todos los datos de entrada.

#### Ejemplo de Uso

> [!Note] Este ejemplo no es el utilizado en las notebooks, sirve de ejemplo para un fácil reconocimiento de cómo usar los salgoritmo sen el futuro, lo mismo va  para tódos los siguientes (En particular para Kohonen se analizó el set de Europa).



```python
from core.kohonen import Kohonen
import numpy as np

# Cargar o crear datos
data = np.random.rand(100, 3)  # Datos de ejemplo con 100 muestras y 3 características

# Crear y ajustar la red de Kohonen
kohonen_net = Kohonen(data, grid_size=5)
kohonen_net.fit(num_iterations=100)

# Mapeo de la BMU (Unidad de Mejor Coincidencia)
bmu_mapping = kohonen_net.bmu_mapping
```

---

### Análisis de Componentes Principales (PCA)

PCA es una técnica estadística para simplificar datos reduciendo su dimensionalidad mientras se conserva la mayor variabilidad posible. Nuestra implementación hace uso de la matriz de covarianzas y no SVD.

#### Ejemplo de Uso

```python
from core.pca import PCA
import pandas as pd

# Cargar datos
data = pd.DataFrame(np.random.rand(100, 4), columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])

# Aplicar PCA
pca = PCA(n_components=2)
pca.fit(data)

# Transformar datos a 2 componentes
data_transformed = pca.transform(data)
```

---

### Regla de Oja

La regla de Oja es un algoritmo de aprendizaje en línea para extraer la primera componente principal de forma iterativa, maximizando la varianza explicada.

#### Ejemplo de Uso

```python
from core.oja import Oja
import numpy as np

# Datos sintéticos
data = np.random.rand(50, 5)  # 50 muestras con 5 características

# Aplicar la regla de Oja
oja_model = Oja(seed=42, num_features=5)
oja_model.fit(data, epochs=100)
```

---

### Regla de Sanger

La regla de Sanger es una extensión de la regla de Oja para encontrar múltiples componentes principales ortogonales.

#### Ejemplo de Uso

```python
from core.sanger import Sanger
import numpy as np

# Generar datos
data = np.random.rand(60, 4)  # 60 muestras, 4 características

# Aplicar la regla de Sanger
sanger_net = Sanger(seed=42, num_features=4, num_components=2)

# Obtener pesos equivalentes a componentes principales
sanger_weights = sanger_net.fit(data, epochs=100)
```

---

### Red de Hopfield

La red de Hopfield es una red neuronal recurrente que actúa como memoria asociativa, capaz de almacenar y recuperar patrones, en este caso usando letras en matrices 5x5.

#### Ejemplo de Uso

Utilizando el conjunto de datos en [data](./data/letters_cool.txt):

```python
from core.hopfield import Hopfield
import numpy as np

# Crear la red de Hopfield
n_neurons = 25
hopfield_net = Hopfield(n_neurons)

# Parseo y aplanamiento de patrones de letras
letters = parse_letters_from_txt("../data/letters_cool.txt")
flattened_letters = {k: v.flatten() for k, v in letters.items()}

# Selección de patrones para almacenar en la red Hopfield
patterns = [flattened_letters[letter] for letter in ['A', 'B', 'C', 'D']]
hopfield_net.train(patterns)

# Patrón de entrada ruidoso
input_pattern = np.array([1, 1, 1, 1, 1, 1,-1, -1, 1, -1, 1,-1, -1, -1, 1, 1,1, -1, 1, 1, 1,-1, -1, -1, 1])

# Recuperación del patrón
output_pattern, state_history = hopfield_net.update(input_pattern)
```

## Análisis

En la carpeta [src](./src/) hay notebooks con un análisis detallado del comportamiento de los algorítmos.

## Referencias

1. **Red de Hopfield**: [Wikipedia](https://es.wikipedia.org/wiki/Red_de_Hopfield)
2. **Red de Kohonen**: [Mapas Autoorganizados](https://es.wikipedia.org/wiki/Mapa_autoorganizado)
3. **PCA y Reducción de Dimensionalidad**: [Análisis de Componentes Principales](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales)
4. **Regla de Oja y Sanger**: [Hebbian Learning Rules](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm)