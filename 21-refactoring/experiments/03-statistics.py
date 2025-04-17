# %%

import statistics

MAIN = "__main__" == __name__

# %%

if MAIN:
    data = [1, 2, 3, 4, 5, 6]
    print(f"Mean: {statistics.mean(data)}")
    print(f"Median: {statistics.median(data)}")
    print(f"Mode: {statistics.mode(data)}")

# %%
