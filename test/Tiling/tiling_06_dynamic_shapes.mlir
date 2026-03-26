// ============================================================================
// Test 6: Matrices with dynamic dimensions (?)
// Expectation: affine.min for dynamic boundary handling
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_dynamic_shapes(
    %A: memref<?x?xf32>,
    %B: memref<?x?xf32>,
    %C: memref<?x?xf32>) {

  linalg.matmul
    ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
    outs(%C: memref<?x?xf32>)

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_dynamic_shapes
// ===========================================================================

// CHECK-DAG: arith.constant 0 : index
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
