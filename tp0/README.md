
# TP0 SIA - Análisis de Datos

## Introducción

Trabajo práctico orientativo para la materia Sistemas de Inteligencia Artificial con el
objetivo de evaluar la función de captura de un Pokemon.

[Enunciado](docs/SIA_TP0.pdf)

### Requisitos

- Python3
- pip3
- [pipenv](https://pypi.org/project/pipenv/)

### Instalación

Parado en la carpeta del tp0 ejecutar

```sh
pipenv install
```

para instalar las dependencias necesarias en el ambiente virtual

## Ejecución

El siguiente comando agarra todos los pokemons en la carpeta `./config` (se pueden agregar más considerando que respetan el formato basico de nombre y pokebolas)

```sh
pipenv run python main.py ./configs
```

>   [!Note]
>   Actualmente está descomentado el ej1 y producirá la salida de los gráficos en el directorio ./output/1a
>   Otros ejercicios generan gráficos de forma semejante, para ejecutarlos descomentar las lineas de invocación en el archivo main.py (buscar ACA!!!)
>   Tener en cuenta que no cualquier combinación de load y analyze es posible ya que hay una correspondencia entre ejercicios. Si se desea ejecutar correctamente ejecutar el par correcto de load y analyze para el ejercicio en particular
