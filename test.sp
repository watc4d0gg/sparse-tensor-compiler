A : Tensor{(i, j) -> Levels{(i): Dense, (j): Dense}}
B : Tensor{(i) -> Levels{(i): Compressed}}
C : Tensor{(i) -> Levels{(i): Compressed}}
A(i, j) = B(i) + C(i + j)