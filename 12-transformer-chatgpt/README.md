# QKV

For example, `What is StatQuest`

Q: `is`
K: `What`, `is`
V: `What`, `is`

Q * K^T: obviously Query `is` is more similar with Key `is`

Let's say
Softmax(Q * K^T): 0.0 for `What`, 1.0 for `is`

Softmax(Q * K^T) * V: 0.0 * Value `What` + 1.0 * Value `is`

So, the attention of `is` is the calculation above.

# Be careful of `torch.nn.CrossEntropyLoss`

```
criterion = torch.nn.CrossEntropyLoss()
# expected input: N, C, d1
# expected output: N, d1
loss = criterion(input, target)
```