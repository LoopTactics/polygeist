#include "MLIRGen.h"
#include "MLIRAffineExprGen.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

namespace teckyl {

// Resursively maps the function `fn` to `tree` and all of its
// descendants in preorder.
static void mapRecursive(const lang::TreeRef &tree,
                         std::function<void(const lang::TreeRef &)> fn) {
  fn(tree);

  for (auto e : tree->trees())
    mapRecursive(e, fn);
}

// Returns a map with one entry per output tensor specifying their
// ranks for the TC definition `def`. If the same tensor is indexed
// with multiple ranks (e.g., C(i, j) = ... and C(i, j, k) = ..., a fatal
// error occurs.
std::unordered_map<std::string, size_t>
MLIRGenImpl::collectOutputRanks(const lang::Def &def) {
  std::set<std::string> outParamNames;
  std::unordered_map<std::string, size_t> ranks;

  for (const lang::Param &outParam : def.returns())
    outParamNames.insert(outParam.ident().name());

  for (const lang::Comprehension &compr : def.statements()) {
    std::string name = compr.ident().name();
    size_t rank = compr.indices().size();

    if (outParamNames.find(name) != outParamNames.end()) {
      auto it = ranks.find(name);

      if (it != ranks.end()) {
        if (it->second != rank) {
          // TODO: add name 'name'
          llvm::report_fatal_error(
              "Multiple ranks found for output tensor for ");
        }
      } else {
        ranks.insert({name, rank});
      }
    }
  }

  return ranks;
}

mlir::FloatType MLIRGenImpl::getFloatType(int kind) {
  switch (kind) {
  case lang::TK_DOUBLE:
    return builder_.getF64Type();
  case lang::TK_FLOAT:
    return builder_.getF32Type();
  case lang::TK_FLOAT16:
    return builder_.getF16Type();
  case lang::TK_FLOAT32:
    return builder_.getF32Type();
  case lang::TK_FLOAT64:
    return builder_.getF64Type();
  default:
    llvm_unreachable("Not a float type");
  }
}

mlir::Type MLIRGenImpl::getScalarType(int kind) {
  switch (kind) {
  case lang::TK_DOUBLE:
  case lang::TK_FLOAT:
  case lang::TK_FLOAT16:
  case lang::TK_FLOAT32:
  case lang::TK_FLOAT64:
    return getFloatType(kind);
  case lang::TK_INT2:
    return builder_.getIntegerType(2);
  case lang::TK_INT4:
    return builder_.getIntegerType(4);
  case lang::TK_INT8:
    return builder_.getIntegerType(8);
  case lang::TK_INT16:
    return builder_.getIntegerType(16);
  case lang::TK_INT32:
    return builder_.getIntegerType(32);
  case lang::TK_INT64:
    return builder_.getIntegerType(64);
  case lang::TK_SIZET:
    return builder_.getIndexType();
  default:
    llvm_unreachable("Unsupported type");
  }
}

mlir::Type MLIRGenImpl::getTensorType(const lang::TensorType &tensorType,
                                      size_t ndims) {
  mlir::Type scalarType = getScalarType(tensorType.scalarType());
  if (ndims > 0)
    return mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(ndims, -1),
                                       scalarType);
  return scalarType;
}

// Collects the set of iterators of a comprehensions by listing all
// identifiers and retaining only those that are not in the symbol
// table `symTab`.
static std::unordered_map<std::string, teckyl::IteratorKind> collectIterators(
    const lang::Comprehension &comprehension,
    const llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symTab) {
  std::unordered_map<std::string, IteratorKind> iterators;

  for (const lang::Ident &lhsIndex : comprehension.indices())
    iterators.emplace(lhsIndex.name(), IteratorKind::LHS);

  mapRecursive(comprehension.rhs(), [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_IDENT) {
      std::string name = lang::Ident(t).name();

      if (iterators.find(name) == iterators.end() && symTab.count(name) == 0) {
        iterators.emplace(name, IteratorKind::RHSOnly);
      }
    }
  });
  return iterators;
}

// FIXME: run sema analysis, and collect TK_ACCESS
static llvm::SmallVector<std::pair<std::string, lang::ListView<lang::TreeRef>>,
                         8>
collectTensorAccessesSeq(const lang::TreeRef &t) {
  llvm::SmallVector<std::pair<std::string, lang::ListView<lang::TreeRef>>, 8>
      res;

  // Collect all tensor accesses in subexpressions
  mapRecursive(t, [&](const lang::TreeRef &e) {
    if (e->kind() == lang::TK_APPLY) {
      llvm::outs() << lang::pretty_tree(e) << "\n";
      lang::Apply a = lang::Apply(e);
      res.push_back(std::make_pair(a.name().name(), a.arguments()));
    }
  });
  return res;
}

mlir::Operation *MLIRGenImpl::buildLinalgReductionCore(
    const lang::Comprehension &c, mlir::Value tensor,
    const std::unordered_map<std::string, IteratorKind> &iterators,
    const llvm::SmallVectorImpl<std::string> &iteratorSeq,
    mlir::Location location) {
  llvm::outs() << __func__ << "\n";
  llvm::outs() << lang::pretty_tree(c.rhs()) << "\n";

  llvm::SmallVector<mlir::Value, 4> inputOperands;
  llvm::SmallVector<std::string, 4> operandsAsString;
  llvm::SmallVector<mlir::Value, 2> outputOperands;
  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps;
  llvm::SmallVector<mlir::Type, 4> types;
  llvm::SmallVector<std::string, 4> tensorIds;

  llvm::SmallVector<std::pair<std::string, lang::ListView<lang::TreeRef>>, 8>
      tensorAccesses = collectTensorAccessesSeq(c.rhs());

  llvm::errs() << "read tensor accesses: " << tensorAccesses.size() << "\n";

  // Mapping between dimension id and schedule dimension.
  std::unordered_map<std::string, size_t> iteratorDims;
  size_t dim = 0;
  for (std::string it : iteratorSeq)
    iteratorDims.emplace(it, dim++);

  MLIRAffineExprGen affGen(builder_.getContext(), iteratorDims);

  // get codomain dimension for affine map.
  size_t codomainDim = 0;
  for (const std::pair<std::string, lang::ListView<lang::TreeRef>> &p :
       tensorAccesses) {
    size_t currentDim = p.second.size();
    codomainDim = std::max(currentDim, codomainDim);
  }

  llvm::errs() << "codomainDim: " << codomainDim << "\n";

  // map tensor name to affine map.
  std::unordered_map<std::string, mlir::AffineMap> tensorToMap;
  // handle inputs.
  for (const std::pair<std::string, lang::ListView<lang::TreeRef>> &access :
       tensorAccesses) {
    llvm::SmallVector<mlir::AffineExpr, 8> affineExprs =
        affGen.buildAffineExpressions(access.second);
    mlir::AffineMap map =
        mlir::AffineMap::get(codomainDim, 0, affineExprs, context_);
    llvm::errs() << "map: "
                 << "\n";
    map.dump();
    mlir::Value memrefValue = symbolTable_.lookup(access.first);
    // convert memref to tensor.
    mlir::Value tensorValue = builder_.create<mlir::memref::TensorLoadOp>(
        builder_.getUnknownLoc(), memrefValue);
    inputOperands.push_back(tensorValue);
    tensorToMap.insert({access.first, map});
    tensorIds.push_back(access.first);
    operandsAsString.push_back(access.first);
  }

  llvm::errs() << "tensor to map: " << tensorToMap.size() << "\n";

  // outputs. TODO: only single output?
  outputOperands.push_back(tensor);
  assert(outputOperands.size() == 1 && "expect a single output");
  llvm::SmallVector<mlir::AffineExpr, 8> affineExprs =
      affGen.buildAffineExpressions(c.indices());
  mlir::AffineMap map =
      mlir::AffineMap::get(codomainDim, 0, affineExprs, context_);
  llvm::errs() << "map: "
               << "\n";
  map.dump();
  tensorToMap.insert({c.ident().name(), map});
  types.push_back(outputOperands[0].getType());
  tensorIds.push_back(c.ident().name());
  operandsAsString.push_back(c.ident().name());

  // iterator types.
  llvm::SmallVector<mlir::StringRef, 4> iteratorTypes;
  for (const std::string it : iteratorSeq) {
    if (iterators.at(it) == IteratorKind::LHS)
      iteratorTypes.push_back("parallel");
    else
      iteratorTypes.push_back("reduction");
  }

  // order the map based on the input.
  for (std::string operand : operandsAsString)
    indexingMaps.push_back(tensorToMap[operand]);

  llvm::errs() << "bulding linalg generic op\n";
  // TODO: do not push tensor output if dealing with =
  mlir::Operation *genericOp = builder_.create<mlir::linalg::GenericOp>(
      builder_.getUnknownLoc(), types, inputOperands, outputOperands,
      indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        assert(tensorIds.size() == args.size() &&
               "tensor ids must be the same as block args");
        // mapping from lang::Tree ids to block arguments.
        std::map<std::string, mlir::Value> valMap;
        int i = 0;
        for (std::string &tensorId : tensorIds)
          valMap.insert({tensorId, args[i++]});

        MLIRMappedValueExprGen gen(context_, valMap, nestedLoc, nestedBuilder);
        // yeild value.
        mlir::Value result;
        // rhs.
        mlir::Value rhsVal = gen.buildExpr(c.rhs());
        // accumulator for the output tensor is always the last value.
        mlir::Value acc = args[args.size() - 1];

        // build reduction operator.
        switch (c.assignment()->kind()) {
        case lang::TK_PLUS_EQ:
        case lang::TK_PLUS_EQ_B:
          result = gen.buildBinaryExprFromValues<mlir::AddFOp, mlir::AddIOp>(
              rhsVal, acc, builder_.getUnknownLoc());
          break;
        case lang::TK_TIMES_EQ:
        case lang::TK_TIMES_EQ_B:
          llvm_unreachable("Unsupported operator");
          break;
        case '=':
          result = rhsVal;
          break;
        default:
          llvm_unreachable("Unsupported operator");
        }

        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, result);
      });
  return genericOp;
}

void MLIRGenImpl::buildTensorInitialization(mlir::Value tensor,
                                            NeutralElement elem) {
  mlir::Type type = tensor.getType();
  assert(type.isa<mlir::TensorType>() && "expect a tensor type");
  mlir::TensorType tensorType = type.dyn_cast<mlir::TensorType>();
  mlir::Type scalarType = tensorType.getElementType();
  mlir::Value constant;
  if (isMLIRFloatType(scalarType)) {
    if (elem == NeutralElement::Zero)
      constant = builder_.create<mlir::ConstantOp>(
          builder_.getUnknownLoc(), scalarType,
          builder_.getFloatAttr(scalarType, 0.0));
    else
      constant = builder_.create<mlir::ConstantOp>(
          builder_.getUnknownLoc(), scalarType,
          builder_.getFloatAttr(scalarType, 1.0));
  } else if (isMLIRIntType(scalarType)) {
    // TODO: Here we generate only i32. Which is wrong.
    if (elem == NeutralElement::Zero)
      constant = builder_.create<mlir::ConstantOp>(
          builder_.getUnknownLoc(), scalarType, builder_.getI32IntegerAttr(0));
    else
      constant = builder_.create<mlir::ConstantOp>(
          builder_.getUnknownLoc(), scalarType, builder_.getI32IntegerAttr(1));
  }
  builder_.create<mlir::linalg::FillOp>(builder_.getUnknownLoc(), constant,
                                        tensor);
}

mlir::Value MLIRGenImpl::buildComprehension(const lang::Comprehension &c) {
  llvm::outs() << lang::pretty_tree(c) << "\n";
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(symbolTable_);

  std::unordered_map<std::string, IteratorKind> iterators =
      collectIterators(c, symbolTable_);
  std::unordered_set<std::string> iteratorSet;
  std::unordered_set<std::string> iteratorSetReduction;

  for (const std::pair<std::string, IteratorKind> &it : iterators) {
    iteratorSet.insert(it.first);
    if (it.second == IteratorKind::RHSOnly)
      iteratorSetReduction.insert(it.first);
  }

  // Decide on an arbitrary order of iterators for the loop nest.
  llvm::SmallVector<std::string, 8> iteratorSeq;
  for (std::pair<std::string, IteratorKind> it : iterators)
    iteratorSeq.push_back(it.first);

  const std::string outTensorName = c.ident().name();
  mlir::Value outMemRefVal = symbolTable_.lookup(outTensorName);
  if (!outMemRefVal) {
    llvm::errs() << "outTensorName: " << outTensorName << "\n";
    assert(outMemRefVal && "outMemRefVal not founded in symbolTable");
  }
  // convert to memref to tensor
  mlir::Value outTensorVal = builder_.create<mlir::memref::TensorLoadOp>(
      builder_.getUnknownLoc(), outMemRefVal);
  symbolTable_.insert(llvm::StringRef(outTensorName), outTensorVal);
  outTensorVal.dump();

  if (c.assignment()->kind() == lang::TK_PLUS_EQ_B)
    buildTensorInitialization(outTensorVal, NeutralElement::Zero);
  else if (c.assignment()->kind() == lang::TK_TIMES_EQ_B)
    buildTensorInitialization(outTensorVal, NeutralElement::One);
  else if (c.assignment()->kind() == lang::TK_MAX_EQ_B ||
           c.assignment()->kind() == lang::TK_MIN_EQ_B) {
    llvm_unreachable("Unsupported reduction");
  }

  // TODO: Assuming we can always generate linalg operations.
  mlir::Operation *redCore = buildLinalgReductionCore(
      c, outTensorVal, iterators, iteratorSeq, builder_.getUnknownLoc());

  // cast back to memref, and introduce a copyOp to avoid DCE.
  mlir::Value cast = builder_.create<mlir::memref::BufferCastOp>(
      builder_.getUnknownLoc(), outMemRefVal.getType(), redCore->getResult(0));
  builder_.create<mlir::linalg::CopyOp>(builder_.getUnknownLoc(), cast,
                                        outMemRefVal);
  return cast;
}

mlir::FuncOp MLIRGenImpl::buildFunction(const std::string name,
                                        const lang::Def &def) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(symbolTable_);
  llvm::SmallVector<mlir::Type, 8> funcArgsTypes;

  for (lang::Param param : def.params()) {
    lang::TensorType tensorType = param.tensorType();
    mlir::Type mlirTensorType =
        getTensorType(tensorType, tensorType.dims().size());
    funcArgsTypes.push_back(mlirTensorType);
  }

  std::unordered_map<std::string, size_t> outputRanks = collectOutputRanks(def);

  for (lang::Param param : def.returns()) {
    lang::TensorType tcTensorType = param.tensorType();
    std::string name = param.ident().name();
    // assert(param.typeisInferred() == false && "do not support range
    // inferance");
    mlir::Type mlirTensorType = getTensorType(tcTensorType, outputRanks[name]);
    funcArgsTypes.push_back(mlirTensorType);
  }

  // build function.
  mlir::FunctionType funcType =
      builder_.getFunctionType(funcArgsTypes, llvm::None);
  mlir::FuncOp funcOp =
      mlir::FuncOp::create(builder_.getUnknownLoc(), name, funcType);
  mlir::SymbolTable::setSymbolVisibility(
      funcOp, mlir::SymbolTable::Visibility::Private);

  // add block for function body.
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  builder_.setInsertionPointToStart(entryBlock);

  // Add input and output tensor to symbol table.
  int i = 0;
  for (lang::Param param : def.params()) {
    mlir::BlockArgument arg = funcOp.getArgument(i++);
    symbolTable_.insert(param.ident().name(), arg);
  }
  for (lang::Param param : def.returns()) {
    mlir::BlockArgument arg = funcOp.getArgument(i++);
    symbolTable_.insert(param.ident().name(), arg);
  }

  // build function body.
  for (const lang::Comprehension &comprehension : def.statements())
    buildComprehension(comprehension);

  builder_.create<mlir::ReturnOp>(builder_.getUnknownLoc());
  return funcOp;
}

mlir::FuncOp buildMLIRFunction(mlir::MLIRContext &context,
                               const std::string name, const lang::Def &tc) {
  // MLIRGenImpl generator(context);
  // return generator.buildFunction(name, tc);
  return nullptr;
}

mlir::Value MLIRMappedValueExprGen::buildConstant(const lang::Const &cst) {
  return nullptr;
}

mlir::Value MLIRMappedValueExprGen::buildCmpExpression(const lang::TreeRef &t) {
  return nullptr;
}

mlir::Value MLIRMappedValueExprGen::buildIdent(const lang::Ident &i) {
  return valMap_[i.name()];
}

mlir::Value MLIRMappedValueExprGen::buildIndexLoadExpr(const lang::Access &a) {
  return buildExpr(a.name());
}

mlir::Value
MLIRMappedValueExprGen::buildTernaryExpression(const lang::TreeRef &t) {
  return nullptr;
}

// Builds a binary MLIR expression from a TC expression. Creates an
// operation of type `FOpTy` if the operands are floats or an
// operation of type `IOpTy` if the operands are integers. If the
// operands have different types or if they are neither integers nor
// floats, an error occurs.
template <typename FOpTy, typename IOpTy>
mlir::Value MLIRMappedValueExprGen::buildBinaryExpr(const lang::TreeRef &t) {
  return buildBinaryExprFromValues<FOpTy, IOpTy>(buildExpr(t->trees().at(0)),
                                                 buildExpr(t->trees().at(1)),
                                                 builder_.getUnknownLoc());
}

mlir::Value MLIRMappedValueExprGen::buildExprImpl(const lang::TreeRef &t) {
  llvm::outs() << lang::pretty_tree(t) << "\n";
  switch (t->kind()) {
  case '+':
    return buildBinaryExpr<mlir::AddFOp, mlir::AddIOp>(t);
  case '-':
    return buildBinaryExpr<mlir::SubFOp, mlir::SubIOp>(t);
  case '*':
    return buildBinaryExpr<mlir::MulFOp, mlir::MulIOp>(t);
  case '/':
    return buildBinaryExpr<mlir::DivFOp, mlir::SignedDivIOp>(t);
  case '?':
    return buildTernaryExpression(t);
  case '<':
  case '>':
  case lang::TK_LE:
  case lang::TK_GE:
  case lang::TK_EQ:
    return buildCmpExpression(t);
  case lang::TK_NUMBER:
  case lang::TK_CONST:
    return buildConstant(lang::Const(t));
  case lang::TK_IDENT:
    return buildIdent(lang::Ident(t));
  case lang::TK_APPLY:
    return buildExpr(lang::Apply(t).name());
    // return buildIndexLoadExpr(lang::Access(t));
  default:
    llvm_unreachable("Unknown tree type\n");
  }
}

mlir::Value MLIRMappedValueExprGen::buildExpr(const lang::TreeRef &t) {
  return buildExprImpl(t);
}

} // end namespace teckyl
