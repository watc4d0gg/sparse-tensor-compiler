A : Tensor{(i, j) -> Levels{(i): Dense, (j): Dense}}
B : Tensor{(i, j) -> Levels{(i): Dense, (j): Dense}}
C : Tensor{(i, j) -> Levels{(i): Dense, (j): Dense}}
D : #CSR
A(i, j) = B(i, k) * C(k, j) * D(i, j)