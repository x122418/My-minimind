import torch
# torch.div
a = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]])
b = torch.tensor([[1, 1, 1], 
                  [2, 2, 2]])
c = torch.div(a, b)
# c = torch.tensor([[1.0, 2.0, 3.0],
#                 [2.0, 2.5, 3.0]])


# torch.repeat_interleave
a = torch.tensor([1,2,3])
b = torch.repeat_interleave(a, repeats=2)
# b = torch.tensor([1, 1, 2, 2, 3, 3])

# torch.argsort
a = torch.tensor([3, 1, 2])
b = torch.argsort(a)  # indexes of the sorted elements
# b = torch.tensor([1, 2, 0])

# torch.bincount
a = torch.tensor([0, 1, 1, 2, 2, 2])
b = torch.bincount(a)
# b = torch.tensor([1, 2, 3])  # counts of each integer in the input tensor