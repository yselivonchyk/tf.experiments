import pandas as pd
import numpy as np
import os
import sys
import glob

files = []
for file in glob.glob("./train_log/*.npy"):
  fname = file.split("train_log/")[1].split(".npy")[0]
  parts = fname.split("__")
  assert len(parts) == 3
  assert parts[2] in ["g", "v"]
  step = int(parts[0])
  v_name = parts[1]
  is_grad = parts[2] == "g"
  print(fname, step, v_name, is_grad)
  files.append(
    {
      "file": fname,
      "var": v_name,
      "is_grad": is_grad,
      "step": is_grad,
      "batchnorm": "_bn_" in fname,
      "is_weight": "_W_"  in fname,
      "data": np.load(file, allow_pickle=True)
    }
  )
  print(files[-1])
