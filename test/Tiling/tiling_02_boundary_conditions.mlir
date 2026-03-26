// ============================================================================
// Test 2: Matrices with dimensions NOT divisible by tile size
// 100x100 with tile size 32 → last tile will be 4x4 (100 % 32 = 4)
// Expectation: Correct boundary handling via affine.min
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_100x100x100(
    %A: memref<100x100xf32>,
    %B: memref<100x100xf32>,
    %C: memref<100x100xf32>) {

  linalg.matmul
    ins(%A, %B: memref<100x100xf32>, memref<100x100xf32>)
    outs(%C: memref<100x100xf32>)

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_100x100x100
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 100 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK-DAG: affine.min

// CHECK: memref.subview
// CHECK: memref.subview
// CHECK: memref.subview

// CHECK: linalg.matmul

// CHECK: }
// CHECK: }
// CHECK: }

// CHECK: return
