# nknnsearch
C code and some utilities for matlab multiple kNN search

# Example

```matlab
mex -v COPTIMFLAGS="-O3" nknnsearch.c

kVal = 20;
N = 100000;
x_train = randn(N,25);

[I, D] = nknnsearch(x_train.', x_train.', kVal);
[I, D] = sortnknn(I, D);
```
