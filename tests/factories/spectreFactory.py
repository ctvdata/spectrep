import numpy as np
import json

# Método auxiliar para crear un spectra.
def create_spectre(size, tp, matrix_size):
    list = []

    for i in range(size):
        print("Generando archivo número: " + str(i) + "...")
        dict = {}
        # Agregamos las llaves correspondientes con su id y el texto.
        dict['id'] = i

        # Creamos la matriz bidimensional aleatoria.
        dict['spectre'] = tp, np.random.rand(matrix_size,matrix_size).tolist()

        list.append(dict)

    # Creamos el archivo de tipo jsonl.
    with open("../data/data_dummy_spectre_{}.jsonl".format(tp), 'w') as f:
        for item in list:
            f.write(json.dumps(item) + "\n")

for tp in ["lexic", "syntactic", "semantic"]:
    create_spectre(100, tp,5)