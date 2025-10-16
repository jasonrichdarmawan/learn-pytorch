# %%

import torch
import torch.nn as nn


class BufferDict(nn.Module):
    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.register_buffer(key, value)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        `nn.Module.load_state_dict` won't create new buffers
        that didn't exist at init. It only loads into
        already-registered names.
        """
        for k in list(state_dict.keys()):
            name = k[len(prefix) :]
            self[name] = state_dict.pop(k)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_P = BufferDict()
        self.register_buffer("lora_A", torch.randn(2, 3))

    def load_lora_P(self, value, adapter_name):
        self.lora_P[adapter_name] = value


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(2)])


model = Model()
model.layers[0].load_lora_P(torch.randn(2, 3), "default")
print(f"{model.layers[0].lora_A=}")
print(f"{model.layers[0].lora_P['default']=}")
print(f"{model.layers[1].lora_A=}")
print(f"{model.layers[1].lora_P=}")

# %%

torch.save(model.state_dict(), "./model.pt")

# %%

model = Model()
model.load_state_dict(torch.load("./model.pt"), strict=False)
print(f"{model.layers[0].lora_A=}")
print(f"{model.layers[0].lora_P['default']=}")
print(f"{model.layers[1].lora_A=}")
print(f"{model.layers[1].lora_P=}")

# %%
