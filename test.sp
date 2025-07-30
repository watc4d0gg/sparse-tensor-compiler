A : Tensor<20, 20>{(i, j) -> Dense{i}, Dense{j}}
B : Tensor<20, 40>{(i, j) -> Dense{i}, Compressed{j}}
C : Tensor<40, 20>{(i, j) -> Dense{i}, Compressed{j}}
A(i, j) = B(i, k) * C(k, j)