import sys
import numpy as np
from my_model import model # <---- import your model here

# load path
path = sys.argv[1]
data = np.load(path)
hash_ = path.split("/")[-1].split(".")[0]

train_x, train_y, test_x = (data[x] for x in list(data.keys()))
print("Train X:", train_x.shape)
print("Test X:", test_x.shape)
print("Train Y:", train_y.shape)

test_y = model(train_x, train_y, test_x)

assert test_y.shape == (24000, 4, 3), "Incorrect shape, please check code!"
np.savez(f"./{hash_}_result.npz", test_y)
