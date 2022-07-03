import numpy as np

txt_path = "checkpoints/jittor/best_iter.txt"

print(np.loadtxt(txt_path, delimiter=',', dtype=int))