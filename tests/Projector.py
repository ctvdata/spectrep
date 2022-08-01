import sys
sys.path.append('..')
from simpsom import SOMNet
import numpy as np
import time

init = time.time()
SOMobj = SOMNet(10, 10, np.random.random(50000000).reshape(5000,10000), PBC=True)

SOMobj.train(epochs=10, start_learning_rate=.01)
print("Training finished in %.02f " % (time.time()-init))