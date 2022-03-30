import numpy as np
import json

# Método auxiliar para crear un spectra.
def create_spectre(size, matrix_size):
    list = []

    for i in range(size):
        print("Generando archivo número: " + str(i) + "...")
        dict = {}
        # Agregamos las llaves correspondientes con su id y el texto.
        dict['id'] = i

        # Creamos la matriz bidimensional aleatoria.
        dict['spectre'] = np.random.rand(matrix_size,matrix_size).tolist()

        list.append(dict)

    # Creamos el archivo de tipo jsonl.
    with open("data_dummy_spectre.jsonl", 'w') as f:
        for item in list:
            f.write(json.dumps(item) + "\n")

create_spectre(100, 5)