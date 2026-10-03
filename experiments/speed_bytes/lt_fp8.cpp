// cuBLASLt FP8 GEMM with selectable A/B scale modes (scalar or outer vector), for a Hopper probe.
// out[M,N] (bf16, row-major) = (x[M,K] fp8 row-major) @ (w[N,K] fp8 row-major)^T, scaled by sx (per
// row of x, or scalar) and sw (per row of w, or scalar). Column-major view used by cuBLASLt:
// D(N x M) = op(A) (N x K) * op(B) (K x M) with A = w (transposed), B = x.
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <map>
#include <tuple>

#define LT_CHECK(expr)                                                                   \
  do {                                                                                   \
    cublasStatus_t st_ = (expr);                                                         \
    TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, #expr " failed: ", static_cast<int>(st_)); \
  } while (0)

namespace {
struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
};
cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) LT_CHECK(cublasLtCreate(&h));
  return h;
}
std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, Plan> plans;

Plan& plan_for(int64_t M, int64_t N, int64_t K, int64_t mode_x, int64_t mode_w, int64_t ws_bytes, const void* pa,
               const void* pb) {
  auto key = std::make_tuple(M, N, K, mode_x, mode_w);
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  Plan p;
  LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  int32_t ma = static_cast<int32_t>(mode_w), mb = static_cast<int32_t>(mode_x);
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ma, sizeof(ma)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mb, sizeof(mb)));
  // Vector scale modes are validated against the scale pointers, so set them before the heuristic query.
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, N));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = static_cast<size_t>(ws_bytes);
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t res{};
  int found = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle(), p.op, p.a, p.b, p.c, p.c, pref, 1, &res, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS && found > 0, "no cuBLASLt algorithm for M=", M, " N=", N, " K=", K,
              " mode_x=", mode_x, " mode_w=", mode_w, " (status ", static_cast<int>(st), ")");
  p.algo = res.algo;
  p.workspace = res.workspaceSize;
  return plans.emplace(key, p).first->second;
}
}  // namespace

torch::Tensor lt_fp8_mm(torch::Tensor x, torch::Tensor w, torch::Tensor sx, torch::Tensor sw, int64_t mode_x,
                        int64_t mode_w, torch::Tensor workspace) {
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn && w.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous());
  int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K);
  auto out = torch::empty({M, N}, x.options().dtype(at::kBFloat16));
  const void* pa = sw.data_ptr();
  const void* pb = sx.data_ptr();
  Plan& p = plan_for(M, N, K, mode_x, mode_w, workspace.numel(), pa, pb);
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb)));
  float alpha = 1.0f, beta = 0.0f;
  LT_CHECK(cublasLtMatmul(handle(), p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, out.data_ptr(), p.c,
                          out.data_ptr(), p.c, &p.algo, workspace.data_ptr(), p.workspace,
                          at::cuda::getCurrentCUDAStream()));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("lt_fp8_mm", &lt_fp8_mm); }
