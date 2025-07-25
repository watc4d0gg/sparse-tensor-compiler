#SparseVector = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>

#DC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0: dense, d1: compressed)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0: compressed, d1: compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1: dense, d0: compressed)
}>

#COO = #sparse_tensor.encoding<{
  map = (i, j) -> (i : compressed(nonunique), j : singleton(soa))
}>

#trait_matmul = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k)>,  // A
    affine_map<(i, j, k) -> (k, j)>,  // B
    affine_map<(i, j, k) -> (i, j)>   // output
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "C(i, j) = sum_k A(i, k) * B(k, j)"
}

// func.func @matmul(
//     %A: tensor<20x30xf64, #CSR>,
//     %B: tensor<30x20xf64, #CSC>
//     // %out: tensor<?x?xf64, #CC>
// ) -> tensor<20x20xf64> {
//   // %c0 = arith.constant 0 : index
//   // %c1 = arith.constant 1 : index
//   %out = tensor.empty() : tensor<20x20xf64>
//   %0 = linalg.generic #trait_matmul
//     ins(%A, %B : tensor<20x30xf64, #CSR>, tensor<30x20xf64, #CSC>)
//     outs(%out: tensor<20x20xf64>) {
//       ^bb0(%a: f64, %b: f64, %o: f64):
//         %mul = arith.mulf %a, %b : f64
//         %add = arith.addf %mul, %o : f64
//         linalg.yield %add : f64
//     } -> tensor<20x20xf64>
//   return %0 : tensor<20x20xf64>
// }

// #BSR = #sparse_tensor.encoding<{
//   map = (i, j) -> (
//     i floordiv 2 : dense,
//     j floordiv 2 : compressed,
//     i mod 2 : dense,
//     j mod 2 : dense)
// }>

#trait_SDDMM = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i,k)>,  // B
    affine_map<(i, j, k) -> (k,j)>,  // C
    affine_map<(i, j, k) -> (i,j)>,  // D
    affine_map<(i, j, k) -> (i,j)>   // A (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel"],
  doc = "S(i,j) += spy[S(i,j)] x SUM_k A(i,k) B(k,j)"
}

func.func @SDDMM_block(%arga: tensor<?x?xf32>,
                         %argb: tensor<?x?xf32>,
                         %argc: tensor<?x?xf32>,
                         %argd: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32> {
  %result = linalg.generic #trait_SDDMM
    ins(%argb, %argc, %argd: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32, #CSR>)
    outs(%arga: tensor<?x?xf32>) {
      ^bb(%b: f32, %c: f32, %d: f32, %a: f32):
          %0 = arith.mulf %b, %c : f32
          %1 = arith.mulf %0, %d : f32
          linalg.yield %1 : f32
    } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// func.func @nested_iterate(%sp : tensor<4x8xf32, #COO>) -> index {
//   %c0 = arith.constant 0 : index
//   // Iterates over the first level of %sp
//   %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0
//       : tensor<4x8xf32, #COO> -> !sparse_tensor.iter_space<#COO, lvls = 0 to 1>
//   sparse_tensor.iterate %it1 in %l1 at (%coord0)
//       : !sparse_tensor.iter_space<#COO, lvls = 0 to 1> {
//     // Iterates over the second level of %sp
//     %l2 = sparse_tensor.extract_iteration_space %sp at %it1 lvls = 1
//         : tensor<4x8xf32, #COO>, !sparse_tensor.iterator<#COO, lvls = 0 to 1>
//        -> !sparse_tensor.iter_space<#COO, lvls = 1 to 2>
//     sparse_tensor.iterate %it2 in %l2 at (%coord1)
//         : !sparse_tensor.iter_space<#COO, lvls = 1 to 2> {
//        vector.print %coord0 : index
//        vector.print %coord1 : index
//     }
//   }
//   return %c0 : index
// }

// func.func @matmuliter(
//     %A: tensor<20x30xf64, #CC>,
//     %B: tensor<30x40xf64, #CC>
//     // %out: tensor<?x?xf64, #CC>
// ) -> tensor<20x40xf64, #CC> {
//   %true = arith.constant true
//   %false = arith.constant false

//   // %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index

//   %out = tensor.empty() : tensor<20x40xf64, #CC>

//   // Extract the first level of A
//   %i = sparse_tensor.extract_iteration_space %A lvls = 0
//       : tensor<20x30xf64, #CC> -> !sparse_tensor.iter_space<#CC, lvls = 0 to 1>

//   // Iterate over the first level of A
//   %0 = sparse_tensor.iterate %it1 in %i at (%i_crd) iter_args (%arg0 = %out)
//       : !sparse_tensor.iter_space<#CC, lvls = 0> 
//       -> tensor<20x40xf64, #CC> {

//     // Perform access pattern expansion
//     %values, %filled, %added, %count = sparse_tensor.expand %arg0
//         : tensor<20x40xf64, #CC> to memref<?xf64>, memref<?xi1>, memref<?xindex>

//     // Extract the second level of A
//     %kA = sparse_tensor.extract_iteration_space %A at %it1 lvls = 1
//         : tensor<20x30xf64, #CC>, !sparse_tensor.iterator<#CC, lvls = 0>
//         -> !sparse_tensor.iter_space<#CC, lvls = 1>

//     // Extract the second level of B
//     %kB = sparse_tensor.extract_iteration_space %B lvls = 0
//         : tensor<30x40xf64, #CC> -> !sparse_tensor.iter_space<#CC, lvls = 0>
    
//     // Coiterate over k in A and B
//     %1 = sparse_tensor.coiterate (%kA, %kB) at (%k_crd) iter_args (%arg1 = %count)
//           : (!sparse_tensor.iter_space<#CC, lvls = 1>,
//               !sparse_tensor.iter_space<#CC, lvls = 0>)
//           -> index
//       // v = A[i, k] * B[k, j]
//       case %it2, %it3 {
//         // Retrieve the value in A at the coordinates [i, k]
//         %a = sparse_tensor.extract_value %A at %it2 : tensor<20x30xf64, #CC>, !sparse_tensor.iterator<#CC, lvls = 1>

//         // %v2 = sparse_tensor.extract_value %t2 at %it3 : index
//         // %v = arith.addi %v1, %v2 : index
//         // %yield = sparse_tensor.insert %v into %arg[%coord]

//         // Extract the second level of B
//         %j = sparse_tensor.extract_iteration_space %B at %it3 lvls = 1
//             : tensor<30x40xf64, #CC>, !sparse_tensor.iterator<#CC, lvls = 0>
//             -> !sparse_tensor.iter_space<#CC, lvls = 1>
        
//         // Iterates over the second level of %sp
//         %2 = sparse_tensor.iterate %it4 in %j at (%j_crd) iter_args (%arg2 = %arg1)
//             : !sparse_tensor.iter_space<#CC, lvls = 1> -> index {
//             // Retrieve the value in B at the coordinates [k, j]
//             %b = sparse_tensor.extract_value %B at %it4 : tensor<30x40xf64, #CC>, !sparse_tensor.iterator<#CC, lvls = 1>
//             %result = arith.mulf %a, %b : f64
//             %prev = memref.load %values[%j_crd] : memref<?xf64>
//             %reduced = arith.addf %prev, %result : f64
//             %was_filled = memref.load %filled[%j_crd] : memref<?xi1>
//             %cond = arith.cmpi eq, %was_filled, %false : i1
//             %forward = scf.if %cond -> (index) {
//               memref.store %true, %filled[%j_crd] : memref<?xi1>
//               memref.store %j_crd, %added[%arg1] : memref<?xindex>
//               %next = arith.addi %arg2, %c1 : index
//               scf.yield %next : index
//             } else {
//               scf.yield %arg2 : index
//             }
//             memref.store %reduced, %values[%j_crd] : memref<?xf64>
//             sparse_tensor.yield %forward : index
//         }

//         sparse_tensor.yield %2 : index
//       }
    
//     %result = sparse_tensor.compress %values, %filled, %added, %1 into %arg0[%i_crd] : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<20x40xf64, #CC>
//     sparse_tensor.yield %result : tensor<20x40xf64, #CC>
//   }

//   return %0 : tensor<20x40xf64, #CC>
// }

// // #trait_affine = {
// //   indexing_maps = [
// //     affine_map<(i, j) -> (i + j)>, // A
// //     affine_map<(i, j) -> (j)>,     // B
// //     affine_map<(i, j) -> (i)>      // output
// //   ],
// //   iterator_types = ["parallel", "reduction"],
// //   doc = "C(i) = A(i + j) * B(j)"
// // }

// // func.func @affine(
// //     %A: tensor<?xf64, #SparseVector>,
// //     %B: tensor<?xf64, #SparseVector>
// //     // %out: tensor<?xf64, #SparseVector>
// // ) -> tensor<?xf64, #SparseVector> {
// //   %i0 = arith.constant 0 : index
// //   %out = tensor.empty(%i0) : tensor<?xf64, #SparseVector>
// //   %result = linalg.generic #trait_affine
// //     ins(%A, %B : tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
// //     outs(%out: tensor<?xf64, #SparseVector>) {
// //       ^bb0(%a: f64, %b: f64, %o: f64):
// //           %f0 = arith.constant 0.0 : f64
// //           %0 = sparse_tensor.reduce %a, %b, %f0 : f64 {
// //             ^bb0(%arg0: f64, %arg1: f64):
// //               %mul = arith.mulf %arg0, %arg1 : f64
// //               sparse_tensor.yield %mul : f64
// //           }
// //           linalg.yield %0 : f64
// //     } -> tensor<?xf64, #SparseVector>
// //   return %result : tensor<?xf64, #SparseVector>
// // }

// // #trait_affine_2 = {
// //   indexing_maps = [
// //     affine_map<(i, j) -> (i, j)>,         // A
// //     affine_map<(i, j) -> (i + 2, j + 3)>, // B
// //     affine_map<(i, j) -> (i, j)>          // out
// //   ],
// //   iterator_types = ["parallel", "parallel"],
// //   doc = "C(i, j) += A(i, j) * B(i + 2, j + 3)"
// // }

// // func.func @affine_2(
// //     %A: tensor<?x?xf64, #CC>,
// //     %B: tensor<?x?xf64>
// //     // %out: tensor<?xf64, #SparseVector>
// // ) -> tensor<?x?xf64, #CC> {
// //   %i0 = arith.constant 0 : index
// //   %i1 = arith.constant 1 : index
// //   %out = tensor.empty(%i0, %i1) : tensor<?x?xf64, #CC>
// //   %result = linalg.generic #trait_affine_2
// //     ins(%A, %B : tensor<?x?xf64, #CC>, tensor<?x?xf64>)
// //     outs(%out: tensor<?x?xf64, #CC>) {
// //       ^bb0(%a: f64, %b: f64, %o: f64):
// //           %0 = sparse_tensor.binary %a, %b : f64, f64 to f64
// //             overlap={
// //               ^bb0(%arg0: f64, %arg1: f64):
// //                 %mul = arith.mulf %arg0, %arg1 : f64
// //                 sparse_tensor.yield %mul : f64
// //             }
// //             left={}
// //             right={}
// //           linalg.yield %0 : f64
// //     } -> tensor<?x?xf64, #CC>
// //   return %result : tensor<?x?xf64, #CC>
// // }

// // // CONCLUSION: compound affine index expressions of form "c * i" or "i + c" are only allowed for dense levels

// // #trait_broadcast = {
// //   indexing_maps = [
// //     affine_map<(i, j) -> (i, j)>, // A
// //     affine_map<(i, j) -> (i)>,    // B
// //     affine_map<(i, j) -> (i, j)>  // out
// //   ],
// //   iterator_types = ["parallel", "parallel"],
// //   doc = "C(i, j) = A(i, j) + B(i)"
// // }

// // func.func @broadcast(
// //     %A: tensor<?x?xf64, #CC>,
// //     %B: tensor<?xf64, #SparseVector>
// //     // %out: tensor<?x?xf64, #CC>
// // ) -> tensor<?x?xf64, #CC> {
// //   %i0 = arith.constant 0 : index
// //   %i1 = arith.constant 1 : index
// //   %out = tensor.empty(%i0, %i1) : tensor<?x?xf64, #CC>
// //   %result = linalg.generic #trait_broadcast
// //     ins(%A, %B : tensor<?x?xf64, #CC>, tensor<?xf64, #SparseVector>)
// //     outs(%out: tensor<?x?xf64, #CC>) {
// //       ^bb0(%a: f64, %b: f64, %o: f64):
// //           %f0 = arith.constant 0.0 : f64
// //           %broadcast = sparse_tensor.reduce %b, %f0, %f0 : f64 {
// //             ^bb0(%arg0: f64, %arg1: f64):
// //               sparse_tensor.yield %arg0 : f64
// //           }
// //           %1 = sparse_tensor.binary %a, %broadcast : f64, f64 to f64
// //             overlap={
// //               ^bb0(%arg0: f64, %arg1: f64):
// //                 %sum = arith.addf %arg0, %arg1 : f64
// //                 sparse_tensor.yield %sum : f64
// //             }
// //             left=identity
// //             right=identity
// //           linalg.yield %1 : f64
// //     } -> tensor<?x?xf64, #CC>
// //   return %result : tensor<?x?xf64, #CC>
// // }

// // #trait_broadcast_multi = {
// //   indexing_maps = [
// //     affine_map<(i, j) -> (i, j)>, // A
// //     affine_map<(i, j) -> (i)>,    // B
// //     affine_map<(i, j) -> (i)>,    // C
// //     affine_map<(i, j) -> (i, j)>  // out
// //   ],
// //   iterator_types = ["parallel", "parallel"],
// //   doc = "C(i, j) = A(i, j) + B(i) + C(i)"
// // }

// // func.func @broadcast_multi(
// //     %A: tensor<?x?xf64, #CC>,
// //     %B: tensor<?xf64, #SparseVector>,
// //     %C: tensor<?xf64, #SparseVector>
// //     // %out: tensor<?x?xf64, #CC>
// // ) -> tensor<?x?xf64, #CC> {
// //   %i0 = arith.constant 0 : index
// //   %i1 = arith.constant 1 : index
// //   %out = tensor.empty(%i0, %i1) : tensor<?x?xf64, #CC>
// //   %result = linalg.generic #trait_broadcast_multi
// //     ins(%A, %B, %C : tensor<?x?xf64, #CC>, tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
// //     outs(%out: tensor<?x?xf64, #CC>) {
// //       ^bb0(%a: f64, %b: f64, %c: f64, %o: f64):
// //           %f0 = arith.constant 0.0 : f64
// //           %broadcast_b = sparse_tensor.reduce %b, %f0, %f0 : f64 {
// //             ^bb0(%arg0: f64, %arg1: f64):
// //               sparse_tensor.yield %arg0 : f64
// //           }
// //           %broadcast_c = sparse_tensor.reduce %c, %f0, %f0 : f64 {
// //             ^bb0(%arg0: f64, %arg1: f64):
// //               sparse_tensor.yield %arg0 : f64
// //           }
// //           %1 = sparse_tensor.binary %a, %broadcast_b : f64, f64 to f64
// //             overlap={
// //               ^bb0(%arg0: f64, %arg1: f64):
// //                 %sum = arith.addf %arg0, %arg1 : f64
// //                 sparse_tensor.yield %sum : f64
// //             }
// //             left=identity
// //             right=identity
// //           %2 = sparse_tensor.binary %1, %broadcast_c : f64, f64 to f64
// //             overlap={
// //               ^bb0(%arg0: f64, %arg1: f64):
// //                 %sum = arith.addf %arg0, %arg1 : f64
// //                 sparse_tensor.yield %sum : f64
// //             }
// //             left=identity
// //             right=identity
// //           linalg.yield %2 : f64
// //     } -> tensor<?x?xf64, #CC>
// //   return %result : tensor<?x?xf64, #CC>
// // }

// // // On broadcasting: use sparse_tensor.reduce with an appropriate identity value