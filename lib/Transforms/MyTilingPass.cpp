#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

#define GEN_PASS_DECL_MYTILINGPASS
#include "MyTiling/Passes.h.inc"
#undef GEN_PASS_DECL_MYTILINGPASS

#define GEN_PASS_DEF_MYTILINGPASS
#include "MyTiling/Passes.h.inc"
#undef GEN_PASS_DEF_MYTILINGPASS

using namespace mlir;

struct TileUsingSCFPattern final: public OpInterfaceRewritePattern<TilingInterface> {
public:
  TileUsingSCFPattern(MLIRContext *ctx, scf::SCFTilingOptions options)
      : OpInterfaceRewritePattern<TilingInterface>(ctx),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("tiled"))
      return failure();

    FailureOr<scf::SCFTilingResult> result =
        scf::tileUsingSCFForOp(rewriter, op, options);
    if (failed(result))
      return failure();

    for (Operation *newOp : result->tiledOps)
      newOp->setAttr("tiled", rewriter.getUnitAttr());

    rewriter.replaceOp(op, result->replacements);
    return success();
  }

private:
  scf::SCFTilingOptions options;
};

struct MyTilingPass final: public ::impl::MyTilingPassBase<MyTilingPass> {
public:
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

  SmallVector<OpFoldResult> tileSizesOFR = llvm::to_vector(
      llvm::map_range(this->tileSizes, [&](int64_t v) -> OpFoldResult {
        return b.getIndexAttr(v);
      }));
  if (tileSizesOFR.empty())
    tileSizesOFR = {b.getIndexAttr(32), b.getIndexAttr(32), b.getIndexAttr(32)};

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOFR);

  RewritePatternSet patterns(ctx);
  patterns.add<TileUsingSCFPattern>(ctx, tilingOptions);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
std::unique_ptr<Pass> createMyTilingPass() {
  return std::make_unique<MyTilingPass>();
}
} // namespace mlir
