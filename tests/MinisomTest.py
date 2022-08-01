import sys
sys.path.append('..')
from minisom.minisom import MiniSom
from spectraltrep.spaceUnification import CorpusReader
import numpy as np
import time
import pickle

init = time.time()

cr = CorpusReader(".\outputs\LexicVectors.jsonl")

data = cr.readFeatureVectors()

model = MiniSom(20, 20, data.shape[1], learning_rate=0.1)
# model.train(data, num_iteration=1000)
with open('som.p', 'wb') as outfile:
    pickle.dump(model, outfile)

# weights = model.get_weights()
# np.save("outputs/pesos.npy",weights)

print("Network trained in %.02f seconds" % (time.time()-init))
