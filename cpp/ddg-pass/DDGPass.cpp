// DDGPass.cpp — Per-basic-block (or virtual-BB) computational DDG emitter.
//
// Minimum LLVM: 15  (new-pass-manager plugin API required)
// Primary targets: LLVM 18, LLVM 19
// Pipeline name:   ddgpass  (use with -passes="ddgpass")
//
// The pass iterates every non-declaration function in the module and emits one
// DOT digraph per basic block (or per virtual-BB group when -ddg-virtual-bb is
// set).  Nodes are compute instructions; edges are def-use data dependences
// within the same group.  Boundary producers and consumers are either
// materialised as external-in nodes (-ddg-emit-input-boundary) or recorded as
// node attributes (ext_in / ext_out), depending on the selected mode.

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Replace non-alphanumeric, non-dash, non-underscore characters with
// underscores so that the result is safe for use as a filename component.
static std::string sanitizeFileName(const std::string &Name) {
  std::string Out;
  Out.reserve(Name.size());
  for (unsigned char C : Name)
    Out.push_back((std::isalnum(C) || C == '_' || C == '-') ? char(C) : '_');
  return Out;
}

// Escape a string for use as a DOT attribute value:
// backslash and double-quote are escaped; newlines and carriage returns become
// spaces.
static std::string dotEscape(StringRef S) {
  std::string Out;
  Out.reserve(S.size());
  for (char c : S) {
    switch (c) {
    case '\\':
      Out += "\\\\";
      break;
    case '\"':
      Out += "\\\"";
      break;
    case '\n':
    case '\r':
      Out += ' ';
      break;
    default:
      Out += c;
      break;
    }
  }
  return Out;
}

// Return the textual IR representation of a constant for use as a
// deduplication key.
static std::string canonicalConstKey(const Constant *C) {
  std::string S;
  raw_string_ostream OS(S);
  C->print(OS);
  return OS.str();
}

// Return a deterministic string key for any Value:
//   instruction  → "inst:<opcode>@<funcname>:<address>"
//   argument     → "arg:<funcname>:<argno>"
//   constant     → "const:<ir-text>"
//   other        → operand text
// The address component of instruction keys gives translation-unit-unique
// identity within a single opt invocation; it is not reproducible across runs
// but is only used for intra-pass graph construction.
static std::string stableKey(const Value *V) {
  if (!V)
    return "v:null";
  std::string S;
  raw_string_ostream OS(S);
  if (const auto *I = dyn_cast<Instruction>(V)) {
    OS << "inst:" << I->getOpcodeName() << "@";
    if (const Function *F = I->getFunction())
      OS << F->getName();
    OS << ":" << (const void *)I;
    return OS.str();
  }
  if (const auto *A = dyn_cast<Argument>(V)) {
    OS << "arg:";
    if (const Function *F = A->getParent())
      OS << F->getName();
    OS << ":" << A->getArgNo();
    return OS.str();
  }
  if (const auto *C = dyn_cast<Constant>(V)) {
    OS << "const:";
    C->print(OS);
    return OS.str();
  }
  V->printAsOperand(OS, /*PrintType=*/false);
  return OS.str();
}

// Convenience overload: cast User* to Value* before delegating.
static inline std::string stableKey(const User *U) {
  return stableKey(static_cast<const Value *>(U));
}

//===----------------------------------------------------------------------===//
// CLI flags
//===----------------------------------------------------------------------===//

static cl::opt<bool>
    IncludeLiteralNodes("ddg-include-literals",
                        cl::desc("Emit literal constants as DDG nodes"),
                        cl::init(true));

static cl::opt<bool> ConstantsAreInput(
    "ddg-constants-are-input",
    cl::desc("Treat constants (immediates) as input boundary"), cl::init(true));

static cl::opt<bool>
    UniqueLiteralNodes("ddg-unique-literals",
                       cl::desc("Merge identical literal values into one node"),
                       cl::init(true));

static cl::opt<std::string>
    OutDirOpt("ddg-out-dir",
              cl::desc("Directory to write .dot files (default: current dir)"),
              cl::value_desc("path"), cl::init(""));

static cl::opt<bool>
    SplitPerBB("ddg-split-bb",
               cl::desc("Emit one .dot per basic block (else per function)"),
               cl::init(true));

static cl::opt<bool> EmitInputBoundary(
    "ddg-emit-input-boundary",
    cl::desc("Materialise external producers as nodes; no external consumer "
             "nodes are emitted"),
    cl::init(false));

// Alias for -ddg-emit-input-boundary.
static cl::opt<bool>
    EmitBoundaryCompat("ddg-emit-boundary", cl::Hidden,
                       cl::desc("Alias for -ddg-emit-input-boundary"),
                       cl::init(false));

// Return true when either input-boundary flag is set.
static inline bool EmitInputs() {
  return EmitInputBoundary || EmitBoundaryCompat;
}

static cl::opt<bool>
    AnnotateIO("ddg-annotate-io",
               cl::desc("Add is_input/is_output booleans to internal nodes"),
               cl::init(true));

static cl::opt<bool> ExcludeMemoryLike(
    "ddg-exclude-mem",
    cl::desc("Exclude load/store/alloca/atomics/fence from the compute set"),
    cl::init(true));

static cl::opt<bool>
    ExcludeCalls("ddg-exclude-calls",
                 cl::desc("Exclude call/invoke from the compute set"),
                 cl::init(true));

static cl::opt<bool> GEPAsCompute(
    "ddg-gep-as-compute",
    cl::desc("Treat getelementptr as a compute node (address-generation)"),
    cl::init(true));

static cl::opt<bool> IncludeCasts(
    "ddg-include-casts",
    cl::desc("Include type-conversion instructions (zext/sext/trunc/bitcast/"
             "ptrtoint/inttoptr) as compute nodes"),
    cl::init(true));

static cl::opt<std::string> VirtualBB(
    "ddg-virtual-bb",
    cl::desc("Group instructions by virtual-BB tag; pass 'orig' to read the "
             "!ddg.orig_bb metadata; 'none' uses the current basic block"),
    cl::init("none"));

static cl::opt<bool> Quiet("ddg-quiet",
                           cl::desc("Suppress per-file 'wrote' messages"),
                           cl::init(false));

//===----------------------------------------------------------------------===//
// Emission predicates
//===----------------------------------------------------------------------===//

// Return true if I is a type-conversion instruction (zero/sign-extend,
// truncate, bitcast, pointer-integer conversions).
static inline bool isCastish(const Instruction *I) {
  return isa<ZExtInst>(I) || isa<SExtInst>(I) || isa<TruncInst>(I) ||
         isa<BitCastInst>(I) || isa<PtrToIntInst>(I) || isa<IntToPtrInst>(I);
}

// Return true if I is a memory-like instruction (load, store, alloca, atomic
// operations, fence, and optionally GEP when -ddg-gep-as-compute is off).
static inline bool isMemoryLike(const Instruction *I) {
  if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<AllocaInst>(I) ||
      isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) || isa<FenceInst>(I))
    return true;
  if (isa<GetElementPtrInst>(I))
    return !GEPAsCompute;
  return false;
}

// Return true if I should appear as a compute node in the DDG.
static inline bool shouldEmitInst(const Instruction *I) {
  if (I->isTerminator() || isa<PHINode>(I) || I->getType()->isVoidTy())
    return false;
  if (ExcludeCalls && isa<CallBase>(I))
    return false;
  if (ExcludeMemoryLike && isMemoryLike(I))
    return false;
  if (!IncludeCasts && isCastish(I))
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

struct DDGPerBBPass : PassInfoMixin<DDGPerBBPass> {
  int GlobalNodeCounter = 0;

  // Per-group (real-BB or virtual-BB) bookkeeping.
  struct Scope {
    std::map<const Instruction *, std::string> InstName;
    std::map<std::string, std::string> NodeLabel;
    std::map<std::string, std::string> NodeKind; // inst|addr|const|external_in
    std::map<std::string, std::string> NodeBB;
    std::map<std::string, std::string> NodeOp;
    std::map<std::string, int> NodeIdx;
    std::set<std::string> Nodes;

    std::map<std::string, int> IsInput;
    std::map<std::string, int> IsOutput;

    // When boundary input nodes are NOT materialised, record ext_in key-sets.
    std::map<std::string, std::set<std::string>> NodeExtIn;
    // ext_out keys are always collected (output boundary nodes are never
    // emitted).
    std::map<std::string, std::set<std::string>> NodeExtOut;

    std::map<const Constant *, std::string> ConstNames;
    std::map<std::string, std::string> ConstKeyToNodeName;

    std::map<std::string, std::string> ExtInKeyToNodeName;

    std::set<std::pair<std::string, std::string>> EdgeSet;
  };

  // Get or create the DDG node name for an internal compute instruction.
  std::string assignInstNode(Scope &S, const Instruction *I,
                             StringRef GroupName) {
    if (auto It = S.InstName.find(I); It != S.InstName.end())
      return It->second;
    std::string N = "n" + std::to_string(GlobalNodeCounter++);
    S.InstName[I] = N;

    std::string Printed;
    raw_string_ostream RSO(Printed);
    I->print(RSO);
    S.NodeLabel[N] = dotEscape(Printed);

    S.NodeKind[N] = isa<GetElementPtrInst>(I) ? "addr" : "inst";
    S.NodeOp[N] = I->getOpcodeName();
    S.NodeBB[N] = GroupName.str();
    S.IsInput[N] = 0;
    S.IsOutput[N] = 0;
    return N;
  }

  // Get or create the DDG node name for a constant value.
  std::string getOrAssignConstNode(Scope &S, const Constant *C,
                                   StringRef GroupName) {
    if (!IncludeLiteralNodes)
      return "";
    if (UniqueLiteralNodes) {
      std::string Key = canonicalConstKey(C);
      if (auto It = S.ConstKeyToNodeName.find(Key);
          It != S.ConstKeyToNodeName.end())
        return It->second;
      std::string N = "c" + std::to_string(GlobalNodeCounter++);
      S.ConstKeyToNodeName[Key] = N;
      S.NodeLabel[N] = dotEscape(Key);
      S.NodeKind[N] = "const";
      S.NodeBB[N] = GroupName.str();
      return N;
    } else {
      if (auto It = S.ConstNames.find(C); It != S.ConstNames.end())
        return It->second;
      std::string N = "c" + std::to_string(GlobalNodeCounter++);
      S.ConstNames[C] = N;
      std::string Key;
      {
        raw_string_ostream OS(Key);
        C->print(OS);
      }
      S.NodeLabel[N] = dotEscape(Key);
      S.NodeKind[N] = "const";
      S.NodeBB[N] = GroupName.str();
      return N;
    }
  }

  // Get or create the DDG node name for an external (cross-group) input
  // producer.
  std::string getOrMakeExternalInNode(Scope &S, StringRef KeyNoPrefix,
                                      StringRef Label, StringRef GroupName) {
    std::string Key = ("in:" + KeyNoPrefix.str());
    if (auto It = S.ExtInKeyToNodeName.find(Key);
        It != S.ExtInKeyToNodeName.end())
      return It->second;
    std::string N = "x" + std::to_string(GlobalNodeCounter++);
    S.ExtInKeyToNodeName[Key] = N;
    S.NodeLabel[N] = dotEscape(Label);
    S.NodeKind[N] = "external_in";
    S.NodeBB[N] = GroupName.str();
    return N;
  }

  // Insert a directed edge U→V into the scope's edge set, deduplicating.
  static inline void add_edge(Scope &S, const std::string &U,
                              const std::string &V) {
    S.EdgeSet.insert({U, V});
  }

  // Serialise the scope as a DOT digraph file under OutDir (or the current
  // directory when OutDir is empty).
  static void writeDotFile(const Scope &S, StringRef FuncName,
                           StringRef GroupName, StringRef OutDir, bool PerBB) {
    SmallString<256> FilePath;
    std::string FileName =
        PerBB ? ("ddg_" + sanitizeFileName(FuncName.str()) + "__bb_" +
                 sanitizeFileName(GroupName.str()) + ".dot")
              : ("ddg_" + sanitizeFileName(FuncName.str()) + ".dot");

    if (OutDir.empty()) {
      FilePath = FileName;
    } else {
      FilePath = OutDir;
      sys::path::append(FilePath, FileName);
    }

    std::error_code EC;
    raw_fd_ostream File(FilePath, EC, sys::fs::CD_CreateAlways);
    if (EC) {
      errs() << "DDGPass: error opening " << FilePath << ": " << EC.message()
             << "\n";
      return;
    }

    File << "digraph \"" << dotEscape(FuncName) << ":" << dotEscape(GroupName)
         << "\" {\n";
    File << "  graph [label=\"func=" << dotEscape(FuncName)
         << ", bb=" << dotEscape(GroupName) << "\"];\n";

    for (const auto &N : S.Nodes) {
      File << "  \"" << N << "\" [label=\"" << S.NodeLabel.at(N) << "\"";
      if (auto it = S.NodeKind.find(N); it != S.NodeKind.end())
        File << ", kind=\"" << it->second << "\"";
      if (auto it = S.NodeBB.find(N); it != S.NodeBB.end())
        File << ", bb=\"" << dotEscape(it->second) << "\"";
      if (auto it = S.NodeOp.find(N); it != S.NodeOp.end())
        File << ", op=\"" << dotEscape(it->second) << "\"";
      if (auto it = S.NodeIdx.find(N); it != S.NodeIdx.end())
        File << ", idx=" << it->second;
      if (AnnotateIO) {
        int in = (S.IsInput.count(N) ? S.IsInput.at(N) : 0);
        int out = (S.IsOutput.count(N) ? S.IsOutput.at(N) : 0);
        File << ", is_input=" << (in ? 1 : 0)
             << ", is_output=" << (out ? 1 : 0);
      }
      if (!EmitInputs()) {
        auto join = [](const std::set<std::string> &SS) {
          std::string R;
          bool first = true;
          for (const auto &k : SS) {
            if (!first)
              R += ";";
            R += dotEscape(k);
            first = false;
          }
          return R;
        };
        if (auto it = S.NodeExtIn.find(N);
            it != S.NodeExtIn.end() && !it->second.empty())
          File << ", ext_in=\"" << join(it->second) << "\"";
      }
      {
        auto join = [](const std::set<std::string> &SS) {
          std::string R;
          bool first = true;
          for (const auto &k : SS) {
            if (!first)
              R += ";";
            R += dotEscape(k);
            first = false;
          }
          return R;
        };
        if (auto it = S.NodeExtOut.find(N);
            it != S.NodeExtOut.end() && !it->second.empty())
          File << ", ext_out=\"" << join(it->second) << "\"";
      }
      File << "];\n";
    }

    for (const auto &E : S.EdgeSet)
      File << "  \"" << E.first << "\" -> \"" << E.second << "\";\n";

    File << "}\n";
    File.close();
    if (!Quiet)
      errs() << "DDGPass: wrote " << FilePath << "\n";
  }

  // Return the group identifier for an instruction: the virtual-BB tag from
  // !ddg.orig_bb metadata when -ddg-virtual-bb=orig is set, otherwise the
  // basic-block name (or a pointer-derived fallback for unnamed blocks).
  static inline std::string groupOf(const Instruction *I) {
    if (VirtualBB == "orig") {
      if (auto *MD = I->getMetadata("ddg.orig_bb")) {
        if (auto *S = dyn_cast<MDString>(MD->getOperand(0)))
          return S->getString().str();
      }
    }
    const BasicBlock *BB = I->getParent();
    if (BB && BB->hasName())
      return BB->getName().str();
    return "bb_" + std::to_string(reinterpret_cast<uintptr_t>(BB));
  }

  // Build per-group DDGs for function F and write them as DOT files to OutDir.
  void buildAndWritePerBB(Function &F, StringRef OutDir) {
    using std::string;
    std::map<string, Scope> G;

    // Pass 1: create internal compute nodes in program order per group.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (!shouldEmitInst(&I))
          continue;
        string g = groupOf(&I);
        Scope &S = G[g];
        string N = assignInstNode(S, &I, g);
        S.NodeIdx[N] = (int)S.NodeIdx.size();
        S.Nodes.insert(N);
      }
    }

    // Pass 2: edges and input classification per group.
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (!shouldEmitInst(&I))
          continue;
        string gSink = groupOf(&I);
        Scope &S = G[gSink];
        const string &Sink = S.InstName[&I];

        for (Use &Op : I.operands()) {
          Value *V = Op.get();

          if (auto *SrcI = dyn_cast<Instruction>(V)) {
            bool sameGroup = (groupOf(SrcI) == gSink);
            if (sameGroup && shouldEmitInst(SrcI)) {
              std::string Src = assignInstNode(S, SrcI, gSink);
              S.Nodes.insert(Src);
              add_edge(S, Src, Sink);
            } else {
              S.IsInput[Sink] = 1;
              std::string keyNP = stableKey(V);
              if (EmitInputs()) {
                std::string lbl =
                    isa<CallBase>(SrcI)
                        ? std::string("ext:call")
                        : (SrcI ? std::string("ext:") + SrcI->getOpcodeName()
                                : std::string("ext:value"));
                std::string X = getOrMakeExternalInNode(S, keyNP, lbl, gSink);
                S.Nodes.insert(X);
                add_edge(S, X, Sink);
              } else {
                S.NodeExtIn[Sink].insert(keyNP);
              }
            }
          } else if (auto *C = dyn_cast<Constant>(V)) {
            if (IncludeLiteralNodes) {
              std::string Src = getOrAssignConstNode(S, C, gSink);
              if (!Src.empty()) {
                S.Nodes.insert(Src);
                add_edge(S, Src, Sink);
              }
            }
            if (ConstantsAreInput) {
              S.IsInput[Sink] = 1;
              if (EmitInputs()) {
                std::string X = getOrMakeExternalInNode(S, stableKey(C),
                                                        "ext:const", gSink);
                S.Nodes.insert(X);
                add_edge(S, X, Sink);
              } else {
                S.NodeExtIn[Sink].insert(stableKey(C));
              }
            }
          } else {
            // Arguments, Undef, and other non-instruction, non-constant values.
            S.IsInput[Sink] = 1;
            std::string keyNP = stableKey(V);
            if (EmitInputs()) {
              std::string X =
                  getOrMakeExternalInNode(S, keyNP, "ext:arg", gSink);
              S.Nodes.insert(X);
              add_edge(S, X, Sink);
            } else {
              S.NodeExtIn[Sink].insert(keyNP);
            }
          }
        }
      }
    }

    // Pass 3: output classification — annotate only (no output boundary nodes).
    for (auto &[g, S] : G) {
      for (auto &KV : S.InstName) {
        const Instruction *I = KV.first;
        const std::string &N = KV.second;

        bool hasUserInside = false;
        bool hasUserOutsideOrNonEmitted = false;

        for (const User *U : I->users()) {
          if (auto *UI = dyn_cast<const Instruction>(U)) {
            bool sameGroup = (groupOf(UI) == g);
            if (sameGroup && shouldEmitInst(UI))
              hasUserInside = true;
            else
              hasUserOutsideOrNonEmitted = true;
          } else {
            hasUserOutsideOrNonEmitted = true;
          }

          if (hasUserOutsideOrNonEmitted)
            S.NodeExtOut[N].insert(stableKey(U));
        }

        if (hasUserOutsideOrNonEmitted || !hasUserInside)
          S.IsOutput[N] = 1;
      }
    }

    // Emit per-group or per-function DOT files.
    if (SplitPerBB) {
      for (auto &[g, S] : G)
        writeDotFile(S, F.getName(), g, OutDirOpt.getValue(), /*PerBB=*/true);
    } else {
      // Aggregate all groups into a single per-function scope.
      Scope Acc;
      for (auto &[g, S] : G) {
        for (auto &p : S.NodeLabel)
          Acc.NodeLabel[p.first] = p.second;
        for (auto &p : S.NodeKind)
          Acc.NodeKind[p.first] = p.second;
        for (auto &p : S.NodeBB)
          Acc.NodeBB[p.first] = p.second;
        for (auto &p : S.NodeOp)
          Acc.NodeOp[p.first] = p.second;
        for (auto &p : S.NodeIdx)
          Acc.NodeIdx[p.first] = p.second;
        for (auto &p : S.IsInput)
          Acc.IsInput[p.first] = p.second;
        for (auto &p : S.IsOutput)
          Acc.IsOutput[p.first] = p.second;
        for (auto &p : S.NodeExtIn)
          Acc.NodeExtIn[p.first].insert(p.second.begin(), p.second.end());
        for (auto &p : S.NodeExtOut)
          Acc.NodeExtOut[p.first].insert(p.second.begin(), p.second.end());
        Acc.Nodes.insert(S.Nodes.begin(), S.Nodes.end());
        Acc.EdgeSet.insert(S.EdgeSet.begin(), S.EdgeSet.end());
      }
      writeDotFile(Acc, F.getName(), F.getName(), OutDirOpt.getValue(),
                   /*PerBB=*/false);
    }
  }

  // Iterate all non-declaration functions in module M and emit their DDGs.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    GlobalNodeCounter = 0;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      buildAndWritePerBB(F, OutDirOpt.getValue());
    }
    return PreservedAnalyses::all();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// New-pass-manager plugin registration
//===----------------------------------------------------------------------===//

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DDGPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "ddgpass") {
                    MPM.addPass(DDGPerBBPass());
                    return true;
                  }
                  return false;
                });
          }};
}
