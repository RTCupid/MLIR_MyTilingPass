// ============================================================================
// Test 9: Matmul inside scf.for loop
// Expectation: Outer loop remains, three inner loops over tiles of size 32
// ============================================================================

// RUN: my-tiling-opt %s -my-tiling="tile-sizes=32,32,32" | FileCheck %s

func.func @matmul_64x64x64_in_scf_for(
    %A: memref<64x64xf32>,
    %B: memref<64x64xf32>,
    %C: memref<64x64xf32>,
    %iter: index) {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %iter step %c1 {
    linalg.matmul
      ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
      outs(%C : memref<64x64xf32>)
  }

  return
}

// ===========================================================================
// CHECK-LABEL: func.func @matmul_64x64x64_in_scf_for
// ===========================================================================

// CHECK-DAG: %c0 = arith.constant 0 : index
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c64 = arith.constant 64 : index
// CHECK-DAG: %c32 = arith.constant 32 : index

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}

// CHECK-DAG:   memref.subview
// CHECK-DAG:   memref.subview
// CHECK-DAG:   memref.subview
// CHECK:       linalg.matmul 

// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: return
