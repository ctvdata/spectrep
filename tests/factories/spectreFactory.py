import numpy as np
import json

# Helper method to create a spectrum.
def create_spectre(size, tp, matrix_size):
    list = []

    for i in range(size):
        print("File Generation number: " + str(i) + "...")
        dict = {}
        # We add the corresponding keys with their id and the text.
        dict['id'] = i

        # We create the random two -dimensional matrix.
        dict['spectre'] = np.random.rand(matrix_size,matrix_size).tolist()

        list.append(dict)

    # We create the JSONL type file.
    with open("./data/data_dummy_spectre_{}.jsonl".format(tp), 'w') as f:
        for item in list:
            f.write(json.dumps(item) + "\n")

for tp in ["lexic", "syntactic", "semantic"]:
    create_spectre(10, tp,4)