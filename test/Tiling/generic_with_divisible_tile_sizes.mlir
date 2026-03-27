// ============================================================================
// Test 8: Matrices with dimensions divisible by tile size (using linalg.generic)
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_64x64x64_generic(
    %A: memref<64x64xf32>,
    %B: memref<64x64xf32>,
    %C: memref<64x64xf32>) {

  linalg.generic {
    indexing_maps = [
        // Indexing maps for matrix multiplication:
        //   C(i, j) += A(i, k) * B(k, j)
        // Iteration space: (i, j, k)
        affine_map<(i, j, k) -> (i, k)>,
        affine_map<(i, j, k) -> (k, j)>,
        affine_map<(i, j, k) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
    outs(%C : memref<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
  }

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_64x64x64_generic
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 64 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK: memref.subview
// CHECK: memref.subview
// CHECK: memref.subview

// CHECK: linalg.generic

// CHECK: }
// CHECK: }
// CHECK: }

// CHECK: return
