
import numpy as np

path = "otoium_model.npz"

with np.load(path) as model:
    for name in model.files:
        data = model[name]

        print(f"\n=== {name} ===")
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)

        if np.issubdtype(data.dtype, np.number):
            print("Min:", data.min())
            print("Max:", data.max())
            print("Mean:", data.mean())
            print("Std:", data.std())
        else:
            print("Values:")
            print(data.tolist())
