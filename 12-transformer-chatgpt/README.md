# Be careful of `torch.nn.CrossEntropyLoss`

```
criterion = torch.nn.CrossEntropyLoss()
# expected input: N, C, d1
# expected output: N, d1
loss = criterion(input, target)
```