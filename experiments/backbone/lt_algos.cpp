// cuBLASLt algorithm enumeration for y[m, n] = x[m, k] @ W[n, k]^T in BF16.
//
// PyTorch's F.linear issues the same product through cuBLAS (cublasGemmEx) or,
// with preferred_blas_library("cublaslt"), through cublasLtMatmul with the first
// heuristic result. This extension asks cuBLASLt for up to `count` heuristic
// algorithms, reports each one's configuration attributes (tile, split-K,
// reduction scheme, stages, ...), and runs any of them, so the microbenchmark can
// time algorithms PyTorch never selects. Column-major view: C[n, m] = op(A) B with
// A = W (k x n, transposed), B = x (k x m), C = y (n x m); FP32 compute and scale.
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#define LT_CHECK(expr)                                                                   \
  do {                                                                                   \
    cublasStatus_t status_ = (expr);                                                     \
    TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS, #expr " failed with status ", int(status_)); \
  } while (0)

namespace {

cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t handle = nullptr;
  if (handle == nullptr) LT_CHECK(cublasLtCreate(&handle));
  return handle;
}

struct Problem {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;

  Problem(int64_t m, int64_t n, int64_t k) {
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
  }
  ~Problem() {
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatmulDescDestroy(op);
  }
};

Problem& problem(int64_t m, int64_t n, int64_t k) {
  static std::map<std::tuple<int64_t, int64_t, int64_t>, std::unique_ptr<Problem>> cache;
  auto& slot = cache[{m, n, k}];
  if (!slot) slot = std::make_unique<Problem>(m, n, k);
  return *slot;
}

// Attributes reported per algorithm; -1 when cuBLASLt does not report one.
const std::vector<std::pair<const char*, cublasLtMatmulAlgoConfigAttributes_t>> kAttrs = {
    {"algo_id", CUBLASLT_ALGO_CONFIG_ID},
    {"tile_id", CUBLASLT_ALGO_CONFIG_TILE_ID},
    {"splitk_num", CUBLASLT_ALGO_CONFIG_SPLITK_NUM},
    {"reduction_scheme", CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},
    {"cta_swizzling", CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING},
    {"custom_option", CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},
    {"stages_id", CUBLASLT_ALGO_CONFIG_STAGES_ID},
    {"inner_shape_id", CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID},
    {"cluster_shape_id", CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID},
};

int64_t read_attr(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  uint64_t value = 0;  // every attribute above fits; cuBLASLt writes its own width
  size_t written = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), &written) !=
      CUBLAS_STATUS_SUCCESS)
    return -1;
  return written == 2 ? int64_t(uint16_t(value)) : written == 4 ? int64_t(uint32_t(value)) : int64_t(value);
}

}  // namespace

// Returns (algos int64 [r, 8] opaque blobs, attrs int64 [r, len(kAttrs) + 1] with the
// required workspace bytes last, waves float64 [r], attribute names).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::string>> heuristics(
    int64_t m, int64_t n, int64_t k, int64_t workspace_bytes, int64_t count) {
  Problem& p = problem(m, n, k);
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = uint64_t(workspace_bytes);
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  std::vector<cublasLtMatmulHeuristicResult_t> results(count);
  int returned = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle(), p.op, p.a, p.b, p.c, p.c, pref, int(count), results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  LT_CHECK(status);
  auto algos = torch::empty({returned, 8}, torch::kInt64);
  auto attrs = torch::empty({returned, int64_t(kAttrs.size()) + 1}, torch::kInt64);
  auto waves = torch::empty({returned}, torch::kFloat64);
  std::vector<std::string> names;
  for (const auto& a : kAttrs) names.emplace_back(a.first);
  names.emplace_back("workspace_bytes");
  for (int i = 0; i < returned; ++i) {
    static_assert(sizeof(results[i].algo.data) == 8 * sizeof(int64_t), "algo blob is 64 bytes");
    std::memcpy(algos[i].data_ptr<int64_t>(), results[i].algo.data, sizeof(results[i].algo.data));
    for (size_t j = 0; j < kAttrs.size(); ++j)
      attrs[i][j] = read_attr(results[i].algo, kAttrs[j].second);
    attrs[i][kAttrs.size()] = int64_t(results[i].workspaceSize);
    waves[i] = double(results[i].wavesCount);
  }
  return {algos, attrs, waves, names};
}

// y = x @ w^T with the given algorithm on the current stream (graph-capturable).
void run(torch::Tensor algo_blob, torch::Tensor x, torch::Tensor w, torch::Tensor y,
         torch::Tensor workspace) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16 &&
                  y.dtype() == torch::kBFloat16, "BF16 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(), "contiguous only");
  TORCH_CHECK(algo_blob.numel() == 8 && algo_blob.dtype() == torch::kInt64 && algo_blob.is_cpu());
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k && y.size(0) == m && y.size(1) == n, "shape mismatch");
  Problem& p = problem(m, n, k);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(algo.data, algo_blob.data_ptr<int64_t>(), sizeof(algo.data));
  const float alpha = 1.0f, beta = 0.0f;
  LT_CHECK(cublasLtMatmul(lt_handle(), p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta,
                          y.data_ptr(), p.c, y.data_ptr(), p.c, &algo, workspace.data_ptr(),
                          size_t(workspace.numel()), at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("heuristics", &heuristics, "cuBLASLt heuristic algorithms for x @ W^T (BF16)");
  mod.def("run", &run, "run x @ W^T with one cuBLASLt algorithm");
}
