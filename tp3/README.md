### Ej1 Perceptrón Simple con Función de Activación Escalón

### Ej2 Perceotrón Simple con Función de Activación Lineal y No Lineal


### Ej3 Perceptrón Multicapa con Función de Activación Sigmoidal

## Ej3 XOR

```json
{
    "problem": {
        "type": "xor",
        "data": "../res/ej3/xor.txt",
        "output": "output/ej3/xor.csv"
    },
    "network": {
        "topology": [2,2,1],
        "activation_function": {
            "method": "sigmoid",
            "beta": 1
        },
        "optimizer": {
            "method": "gradient_descent"
        }
    },
    "training": {
        "seed": 23,
        "epochs": 10000,
        "mini_batch_size": 4,
        "learning_rate": 8,
        "epsilon": 0.01
    }
  
}
```

## Ej3 Parity


## Ej3 Digit discrimination

El archivo configuración para ejecutar el entrenamiento del perceptrón tiene el siguiente formato:

```json
{
    "problem": {
        "type": "number_identifier",
        "data": "../res/ej3/TP3-ej3-digitos.txt",
        "testing_data": "../res/ej3/TP3-ej3-digitos.txt"
    },
    "network": {
        "topology": [35,20,10],
        "activation_function": {
            "method": "sigmoid",
            "beta": 1
        },
        "optimizer": {
            "method": "gradient_descent"
        }
    },
    "training": {
        "seed": 23,
        "epochs": 1000,
        "mini_batch_size": 5,
        "learning_rate": 8,
        "path_to_weights_and_biases": "output/ej3/optional_argument.txt",
        "epsilon": 0.01
    }
}
```

La salida del programa es un archivo con los pesos, bias y resultado de los tests hechos. Si se quiere utilizar los pesos y bias obtenidos en un entrenamiento previo, se puede pasar el path al archivo con los pesos y bias en el campo `path_to_weights_and_biases`. Si no se pasa este campo, se inicializan los pesos y bias de manera aleatoria.

Se puede generar ruido ejecutando:

```python
python3 ej3_gen_noisy_nums.py
```

Se puede cambiar la distribución del ruido generado editando el archivo `tp3/config/ej3_noise.json`. El siguiente es un ejemplo de configuración para generar ruido gaussiano con media 0 y desvío 0.1, para un archivo de 100 números de 7x5 bits. El programa espera los números del 0 al 9 repetidos tantas veces como se quiera, como se muestra en el archivo `tp3/res/ej3/big/100_xor_mean_0_stddev_0.4.txt`.
El parámetro output_xor indica la salida del archivo con los números con ruido, y output_noise indica la salida del archivo con el ruido (sin aplicar a los números). Este último archivo se utiliza para xorear con los dígitos originales y obtener los dígitos con ruido.

```json
{
    "noise": "salt_and_pepper",
    "output_noise": "../res/ej3/big/100_noise_mean_0_stddev_0.4.txt",
    "output_xor": "../res/ej3/big/100_xor_mean_0_stddev_0.4.txt",
    "nums_path": "../res/ej3/big/300_nums.txt",

    "mean": 0,
    "stddev": 0.2,
    "numbers": 100
}
```

## Opciones default

# Entrenamiento y testeo sin ruido
Se puede ejecutar esta opción corriendo en el directorio src el comando:

```python
python ej3.py ../config/ej3_digit_train_clean_clean.json
```

Esto generará un output el directorio `output/ej3/clean_clean`. Luego se puede generar el gráfico ejecutando:
    
```python
python ej3_graphics.py clean
```

Y el path para el gráfico será `tp3/src/output/ej3/digit_accuracy_vs_epochs_clean_clean.png`.

# Entrenamiento sin ruido y testeo con ruido

Se puede ejecutar esta opción corriendo en el directorio src el comando:

```python
python ej3.py ../config/ej3_digit_train_clean_noisy1.json
```

El parámetro testing_data indica el path al archivo que contiene los números con ruido. Este mismo se puede generar con el script `ej3_gen_noisy_nums.py` como se explicó anteriormente.

Esto generará dos output el directorio `output/ej3/clean_noisy1`: uno json para los resultados del training y otro para los resultados de los tests. Luego se puede generar el gráfico de accuracy vs epochs ejecutando:

```python
python ej3_graphics.py clean_noisy1
```

Y el path para el gráfico será `tp3/src/output/ej3/digit_accuracy_vs_epochs_clean_noisy1.png`.

# Entrenamiento con ruido y testeo con ruido

Se puede ejecutar esta opción corriendo en el directorio src el comando:

```python
python ej3.py ../config/ej3_digit_train_noisy1_noisy2.json
```

Ahora el parámetro data indica el path al archivo que contiene los números con ruido. Este mismo se puede generar con el script `ej3_gen_noisy_nums.py` como se explicó anteriormente.
Además, se está utilizando el parámetro path_to_weights_and_biases para utilizar los pesos y bias obtenidos en el entrenamiento con números sin ruido.

Esto generará dos output el directorio `output/ej3/noisy1_noisy2`: uno json para los resultados del training y otro para los resultados de los tests. Luego se puede generar el gráfico de accuracy vs epochs ejecutando:

```python
python ej3_graphics.py noisy1_noisy2
```

Y el path para el gráfico será `tp3/src/output/ej3/digit_accuracy_vs_epochs_noisy1_noisy2.png`.

Ahora se puede generar el gráfico de precisión para cada dígitos ejecutando:

```python
python ej3_graphics.py precision
```

Y el path para el gráfico será `tp3/src/output/ej3/training_with_salt_and_pepper_cross_val.png`.

# Entrenamiento con ruido salt and pepper en base a lo aprendido con ruido gaussiano

Se puede ejecutar esta opción corriendo en el directorio src el comando:

```python
python ej3.py ../config/ej3_digit_second_train_salt_papper.json
```

Ahora el parámetro data indica el path al archivo que contiene los números con ruido salt and pepper. Este mismo se puede generar con el script `ej3_gen_noisy_nums.py` como se explicó anteriormente, utilizando `"noise": "salt_and_pepper"` en el archivo de configuración.
Además, se está utilizando el parámetro path_to_weights_and_biases para utilizar los pesos y bias obtenidos en el entrenamiento con números con ruido gaussiano.

Esto generará dos output el directorio `output/ej3/noisy1_noisy2`: uno json para los resultados del training y otro para los resultados de los tests. Luego se puede generar el gráfico de accuracy vs epochs ejecutando:

```python
python ej3_graphics.py second_train_salt_pepper
```

Y el path para el gráfico será `tp3/src/output/ej3/training_with_salt_and_pepper_cross_val.png`.
