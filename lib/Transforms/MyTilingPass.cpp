#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include <algorithm>
#include <memory>
#include <numeric>

#define GEN_PASS_DECL_MYTILINGPASS
#include "MyTiling/Passes.h.inc"
#undef GEN_PASS_DECL_MYTILINGPASS

#define GEN_PASS_DEF_MYTILINGPASS
#include "MyTiling/Passes.h.inc"
#undef GEN_PASS_DEF_MYTILINGPASS

using namespace mlir;

struct MyTilingPass : public ::impl::MyTilingPassBase<MyTilingPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyTilingPass)

  MyTilingPass() = default;
  MyTilingPass(const MyTilingPass &other)
      : MyTilingPassBase<MyTilingPass>(other) {}

  void runOnOperation() override;
};

void MyTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  Builder b(ctx);

  SmallVector<OpFoldResult> tileSizesOFR =
      llvm::to_vector(llvm::map_range(this->tileSizes, [&](int64_t v) -> OpFoldResult {
        return b.getIndexAttr(v);
      }));
  if (tileSizesOFR.empty())
    tileSizesOFR = {b.getIndexAttr(32), b.getIndexAttr(32), b.getIndexAttr(32)};

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOFR);

  SmallVector<TilingInterface> worklist;
  func.walk([&](TilingInterface op) {
    worklist.push_back(op);
  });

  IRRewriter rewriter(ctx);
  SmallVector<std::pair<TilingInterface, FailureOr<scf::SCFTilingResult>>> results;
  results.resize(worklist.size());

  std::transform(worklist.begin(), worklist.end(), results.begin(),
      [&](TilingInterface &op) -> std::pair<TilingInterface, FailureOr<scf::SCFTilingResult>> {
    if (op->getBlock() == nullptr) return std::make_pair(op, failure());

    return std::make_pair(op, scf::tileUsingSCFForOp(rewriter, op, tilingOptions));
  });

  for (auto &result : results) {
    if (succeeded(result.second)) {
      rewriter.replaceOp(result.first, result.second->replacements);
    }
  }
}

namespace mlir {
std::unique_ptr<Pass> createMyTilingPass() {
  return std::make_unique<MyTilingPass>();
}
} // namespace mlir

