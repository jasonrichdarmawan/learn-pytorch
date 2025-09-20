# %%

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# %%

import torch

from util.runningstats import (
    CombinedStat,
    Mean,
    Quantile,
    tally,
)

# %%

class MyDataSet:
    def __init__(self):
        self.ds = torch.arange(0, 1000 * 1000, dtype=torch.float32).reshape(1000, 1000)

    def __getitem__(self, idx):
        return [self.ds[idx]]

    def __len__(self):
        return 1000


cs = CombinedStat(m=Mean(), q=Quantile())
for [b] in tally(
    cs,
    MyDataSet(),
    cache=None,
    sample_size=10,
    batch_size=100,
):
    print(b.shape)
    cs.add(b)
print(cs.m.mean())
print(cs.q.median())