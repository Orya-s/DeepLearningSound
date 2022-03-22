import numpy as np
import pandas as pd
import pickle

objects = []
with (open("data.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print(len(objects))
print(type(objects))

inner_list = objects[0]  # list - one argument
print(type(inner_list))
print(len(inner_list))  # 100 sets of 3 arg each

one_rec = inner_list[0]  # set of 3
#print(one_rec)
print(len(one_rec))

for x in one_rec:
    print(x)
