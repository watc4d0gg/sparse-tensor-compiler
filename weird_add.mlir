// #SparseVector = #sparse_tensor.encoding<{
//   map = (i) -> (i : compressed)
// }>

// #DC = #sparse_tensor.encoding<{
//   map = (d0, d1) -> (d0: dense, d1: compressed)
// }>

// #CC = #sparse_tensor.encoding<{
//   map = (d0, d1) -> (d0: compressed, d1: compressed)
// }>

// #trait_weird_add = {
//   indexing_maps = [
//     affine_map<(i, j) -> (i, j)>, // A
//     affine_map<(i, j) -> (i, j)>, // B
//     affine_map<(i, j) -> (i)>,    // C
//     affine_map<(i, j) -> (i)>     // out
//   ],
//   iterator_types = ["parallel", "reduction"],
//   doc = "D(i) = sum_j A(i, j) + B(i, j) + C(i)"
// }

// func.func @weird_add(
//     %A: tensor<?x?xf64, #CC>,
//     %B: tensor<?x?xf64, #CC>,
//     %C: tensor<?xf64, #SparseVector>
//     // %out: tensor<?xf64, #SparseVector>
// ) -> tensor<?xf64, #SparseVector> {
//   %i0 = arith.constant 0 : index
//   %out = tensor.empty(%i0) : tensor<?xf64, #SparseVector>
//   %0 = linalg.generic #trait_weird_add
//     ins(%A, %B, %C : tensor<?x?xf64, #CC>, tensor<?x?xf64, #CC>, tensor<?xf64, #SparseVector>)
//     outs(%out: tensor<?xf64, #SparseVector>) {
//       ^bb0(%a: f64, %b: f64, %c: f64, %o: f64):
//           %0 = arith.addf %a, %b : f64
//           %1 = arith.addf %0, %c : f64
// 					%2 = arith.addf %1, %o : f64
//           linalg.yield %2 : f64
//     } -> tensor<?xf64, #SparseVector>
//   return %0 : tensor<?xf64, #SparseVector>
// }

#map = affine_map<(d0, d1) -> (d0 * 3 + 2, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#sparse1 = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
#sparse2 = #sparse_tensor.encoding<{ map = (d0) -> (d0 : dense) }>
module {
  func.func @kernels(%arg0: tensor<?x?xf64, #sparse>, %arg1: tensor<?x?xf64, #sparse>, %arg2: tensor<?xf64, #sparse1>) -> tensor<?xf64, #sparse2> {
    %c0 = arith.constant 0 : index
    %0 = tensor.empty(%c0) : tensor<?xf64, #sparse2>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf64, #sparse>, tensor<?x?xf64, #sparse>, tensor<?xf64, #sparse1>) outs(%0 : tensor<?xf64, #sparse2>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %2 = arith.addf %in, %in_0 : f64
      %cst = arith.constant 0.000000e+00 : f64
      %3 = sparse_tensor.reduce %2, %cst, %cst : f64 {
      ^bb0(%arg3: f64, %arg4: f64):
        sparse_tensor.yield %arg3 : f64
      }
      %4 = arith.addf %3, %in_1 : f64
      %5 = arith.addf %4, %out : f64
      linalg.yield %5 : f64
    } -> tensor<?xf64, #sparse2>
    return %1 : tensor<?xf64, #sparse2>
  }
}