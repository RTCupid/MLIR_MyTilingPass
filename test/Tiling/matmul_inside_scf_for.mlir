// ============================================================================
// Test 9: Matmul inside scf.for loop
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
