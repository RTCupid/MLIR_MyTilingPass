// ============================================================================
// Test 7: Rectangular matrices with different dimensions (M ≠ N ≠ K)
// A: 50x80, B: 80x30 → C: 50x30
// Expectation: Tiling applied correctly to all three dimensions
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=16,16,32" | FileCheck %s

func.func @matmul_rectangular(
    %A: memref<50x80xf32>,
    %B: memref<80x30xf32>,
    %C: memref<50x30xf32>) {

  linalg.matmul
    ins(%A, %B: memref<50x80xf32>, memref<80x30xf32>)
    outs(%C: memref<50x30xf32>)

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_rectangular
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 50 : index
// CHECK-DAG: arith.constant 80 : index
// CHECK-DAG: arith.constant 30 : index
// CHECK-DAG: arith.constant 16 : index
// CHECK-DAG: arith.constant 32 : index

// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK-DAG: affine.min

// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview

// CHECK-DAG: linalg.matmul

// CHECK: return
