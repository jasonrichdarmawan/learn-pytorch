# %%

from collections import Counter

MAIN = "__main__" == __name__

# %%

if MAIN:
    c = Counter(['apple', 'red', 'apple', 'blue', 'red', 'red'])
    print(c)
    print(c['red'])

# %%
