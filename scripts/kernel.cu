#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#include <rocwmma/rocwmma.hpp>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

using rocwmma::bfloat16_t;
using rocwmma::float16_t;
using rocwmma::float32_t;

#define USE_HALF 1

#if USE_HALF

#define ComputeType float16_t
#define AT_PTR_TYPE at::Half
#define TORCH_DTYPE torch::kFloat16
#else

#define ComputeType bfloat16_t
#define AT_PTR_TYPE at::BFloat16
#define TORCH_DTYPE torch::kBFloat16

#endif

const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;
const int ROCWMMA_K = 16;

const int N_WAVES = 32;
const int WAVE_SIZE = 32;

//================================ Matrix multiplication ===============================
// C = k*(A^T)B + C
__device__ void mul_add_kAT_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k,
    const float scale)
{
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragA;
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = (threadIdx.x / WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = wave_id + wave_off * N_WAVES;

        int wave_x = wave_xy % (n / ROCWMMA_N);
        int wave_y = wave_xy / (n / ROCWMMA_N);

        int blk_x = wave_x * ROCWMMA_N;
        int blk_y = wave_y * ROCWMMA_M;

        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K)
            {
                rocwmma::load_matrix_sync(fragA, A + (i * m + blk_y), m);
                rocwmma::load_matrix_sync(fragB, B + (i * n + blk_x), n);
                rocwmma::mma_sync(fragACC, fragA, fragB, fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * n + blk_x), n, rocwmma::mem_row_major);
#pragma unroll 8
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = scale * fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    __syncthreads();
}

// C = k*A(B^T)
__device__ void mul_kA_BT(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k,
    const float scale)
{
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA;
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragB;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = (threadIdx.x / WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = wave_id + wave_off * N_WAVES;

        int wave_x = wave_xy % (n / ROCWMMA_N);
        int wave_y = wave_xy / (n / ROCWMMA_N);

        int blk_x = wave_x * ROCWMMA_N;
        int blk_y = wave_y * ROCWMMA_M;

        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K)
            {
                rocwmma::load_matrix_sync(fragA, A + (blk_y * k + i), k);
                rocwmma::load_matrix_sync(fragB, B + (blk_x * k + i), k);
                rocwmma::mma_sync(fragACC, fragA, fragB, fragACC);
            }
#pragma unroll 8
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = scale * fragACC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    __syncthreads();
}

// C = k*AB + C
__device__ void mul_add_kA_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    const int m, const int n, const int k, float32_t scale)
{

    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA;
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = (threadIdx.x / WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = wave_id + wave_off * N_WAVES;

        int wave_x = wave_xy % (n / ROCWMMA_N);
        int wave_y = wave_xy / (n / ROCWMMA_N);

        int blk_x = wave_x * ROCWMMA_N;
        int blk_y = wave_y * ROCWMMA_M;
        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K)
            {
                rocwmma::load_matrix_sync(fragA, A + (blk_y * k + i), k);
                rocwmma::load_matrix_sync(fragB, B + (i * n + blk_x), n);
                rocwmma::mma_sync(fragACC, fragA, fragB, fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * n + blk_x), n, rocwmma::mem_row_major);
#pragma unroll 8
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = scale * fragACC.x[i] + fragC.x[i];
            }
            rocwmma::store_matrix_sync(C + (blk_y * n + blk_x), fragC, n, rocwmma::mem_row_major);
        }
    }
    __syncthreads();
}

// =========================================================================================

__global__ void
fwd_kernel(
    ComputeType *__restrict__ q,
    ComputeType *__restrict__ k,
    ComputeType *__restrict__ v,
    ComputeType *__restrict__ o,
    float *__restrict__ L,
    const int Tr, const int Tc, const int Br, const int Bc,
    const int nq, const int nkv,
    const int d,
    const float32_t scale,
    const bool causal
)
{
    const int q_offset = (blockIdx.x + blockIdx.y * gridDim.x) * nq * d;
    const int kv_offset = (blockIdx.x + blockIdx.y * gridDim.x) * nkv * d;
    const int L_offset = (blockIdx.x + blockIdx.y * gridDim.x) * nq;

    const int Tr_i = blockIdx.z;
    const int tx = threadIdx.x;

    extern __shared__ ComputeType sram[];
    ComputeType *__restrict__ Si = &sram[0];                // Br * Bc
    ComputeType *__restrict__ Oi = &sram[Br * Bc];          // Br * d
    ComputeType *__restrict__ Qi = &sram[Br * Bc + Br * d]; // Br * d

    if (tx < d)
    {
#pragma unroll 32
        for (int i = 0; i < Br; i++)
        {
            Qi[i * d + tx] = q[q_offset + Tr_i * Br * d + i * d + tx];
            Oi[i * d + tx] = 0;
        }
    }

    __syncthreads();

    // ComputeType *__restrict__ Qi = &q[q_offset + Tr_i * Br * d];
    // ComputeType *__restrict__ Oi = &o[q_offset + Tr_i * Br * d];

    float32_t row_max_old = -INFINITY;
    float32_t l_i = 0;

    for (int j = 0; j < Tc; j++)
    {

        ComputeType *__restrict__ Kj = &k[kv_offset + j * Bc * d];
        ComputeType *__restrict__ Vj = &v[kv_offset + j * Bc * d];

        mul_kA_BT(Qi, Kj, Si, Br, Bc, d, scale);

        if(causal)
        {
            int ele_y = Tr_i * Br;
            int ele_x = j * Bc;
            if ((ele_y < ele_x + Bc - 1) && (tx < Br))
            {
#pragma unroll 32
                for(int i = 0; i < Bc; i++)
                {
                    if ( i >= tx + (ele_y - ele_x + 1)) 
                    #if USE_HALF
                        Si[tx * Bc + i] = -65500.0;
                    #else
                        Si[tx * Bc + i] = -INFINITY;
                    #endif
                }
            }
            __syncthreads();
        }

        if (tx < Br)
        {
            float32_t row_max_new = -INFINITY;
            float32_t row_sum = 0;
#pragma unroll 32
            for (int x = 0; x < Bc; x++)
            {
                row_max_new = max(row_max_new, Si[(tx * Bc) + x]);
            }
            row_max_new = max(row_max_old, row_max_new);
#pragma unroll 32
            for (int x = 0; x < Bc; x++)
            {
                Si[(tx * Bc) + x] = __expf(Si[(tx * Bc) + x] - row_max_new);
                row_sum += Si[(tx * Bc) + x];
            }

            l_i = __expf(row_max_old - row_max_new) * l_i + row_sum;

#pragma unroll 32
            for (int i = 0; i < d; i++)
            {
                Oi[(tx * d) + i] = Oi[(tx * d) + i] * __expf(row_max_old - row_max_new);
            }
            row_max_old = row_max_new;
        }

        __syncthreads();

        mul_add_kA_B(Si, Vj, Oi, Br, d, Bc, 1.0);
    }

    if (tx < Br)
    {
#pragma unroll 32
        for (int i = 0; i < d; i++)
            Oi[tx * d + i] = Oi[tx * d + i] / l_i;
#pragma unroll 32
        for (int i = 0; i < d; i++)
            o[q_offset + Tr_i * Br * d + tx * d + i] = Oi[tx * d + i];

        l_i = row_max_old + __logf(l_i);
        L[L_offset + Tr_i * Br + tx] = l_i;
    }
}
// =================================================================================

#define GRAD_SCALE 0.95

__global__ void
bwd_kernel(
    ComputeType *__restrict__ q,  // [(b*h) x N x d]
    ComputeType *__restrict__ k,  // [(b*h) x N x d]
    ComputeType *__restrict__ v,  // [(b*h) x N x d]
    ComputeType *__restrict__ O,  // [(b*h) x N x d]
    ComputeType *__restrict__ dO, // [(b*h) x N x d]
    ComputeType *__restrict__ dQ, // [(b*h) x N x d]
    ComputeType *__restrict__ dK, // [(b*h) x N x d]
    ComputeType *__restrict__ dV, // [(b*h) x N x d]
    float *__restrict__ Di,       // [(b*h) * N]
    float *__restrict__ L,        // [(b*h) * N]
    const int Tr, const int Tc,
    const int Br, const int Bc,
    const int nq, const int nkv,
    const int d,
    const float32_t scale,
    const bool causal)

{

    const int q_offset = (blockIdx.x + blockIdx.y * gridDim.x) * nq * d;
    const int kv_offset = (blockIdx.x + blockIdx.y * gridDim.x) * nkv * d;
    const int L_offset = (blockIdx.x + blockIdx.y * gridDim.x) * nq;

    const int Tc_j = blockIdx.z;
    const int tx = threadIdx.x;

    extern __shared__ ComputeType sram[];
    ComputeType *__restrict__ Si = &sram[0]; //[Br x Bc]
    ComputeType *__restrict__ Pi = &sram[0];
    ComputeType *__restrict__ dSi = &sram[0];
    ComputeType *__restrict__ dPi = &sram[Br * Bc]; //[Br x Bc]

    ComputeType *__restrict__ Kj = &k[kv_offset + Tc_j * Bc * d]; // [Bc x d]
    ComputeType *__restrict__ Vj = &v[kv_offset + Tc_j * Bc * d]; // [Bc x d]

    ComputeType *__restrict__ dKj = &dK[kv_offset + Tc_j * Bc * d]; // [Bc x d]
    ComputeType *__restrict__ dVj = &dV[kv_offset + Tc_j * Bc * d]; // [Bc x d]

    for (int n_batch = 0; n_batch < ((nq + (blockDim.x - 1)) / blockDim.x); n_batch++)
    {
        int Di_off = n_batch * (blockDim.x) + tx;
        if (Di_off < nq)
        {
            float32_t val = 0;
#pragma unroll 32
            for (int i = 0; i < d; i++)
            {
                val += (dO[q_offset + Di_off * d + i] * O[q_offset + Di_off * d + i]);
            }
            Di[L_offset + Di_off] = val;
        }
    }
    __syncthreads();

    for (int Tr_i = 0; Tr_i < Tr; Tr_i++)
    {
        ComputeType *__restrict__ Qi = &q[q_offset + Tr_i * Br * d];   // [Br x d]
        ComputeType *__restrict__ Oi = &O[q_offset + Tr_i * Br * d];   // [Br x d]
        ComputeType *__restrict__ dOi = &dO[q_offset + Tr_i * Br * d]; // [Br x d]
        ComputeType *__restrict__ dQi = &dQ[q_offset + Tr_i * Br * d]; // [Br x d]
        float32_t *__restrict__ Li = &L[L_offset + Tr_i * Br];         // [Br]
        float32_t *__restrict__ Di_i = &Di[L_offset + Tr_i * Br];

        mul_kA_BT(Qi, Kj, Si, Br, Bc, d, scale); // Qi[Br x d] Kj[Bc x d]

        if(causal)
        {
            int ele_y = Tr_i * Br;
            int ele_x = Tc_j * Bc;
            if ((ele_y < ele_x + Bc - 1) && (tx < Br))
            {
#pragma unroll 32
                for(int i = 0; i < Bc; i++)
                {
                    if ( i >= tx + (ele_y - ele_x + 1)) 
                    #if USE_HALF
                        Si[tx * Bc + i] = -65500.0;
                    #else
                        Si[tx * Bc + i] = -INFINITY;
                    #endif
                }
            }
            __syncthreads();
        }

        if (tx < Br)
        {
#pragma unroll 32
            for (int i = 0; i < Bc; i++)
            {
                Pi[tx * Bc + i] = __expf(Si[tx * Bc + i] - Li[tx]); // Pi [Br x Bc]
            }
        }
        __syncthreads();

        mul_add_kAT_B(Pi, dOi, dVj, Bc, d, Br, 1.0 * GRAD_SCALE); // Pi[Br x Bc] dOi[Br x d]
        mul_kA_BT(dOi, Vj, dPi, Br, Bc, d, 1.0);                  // dPj:[Br x Bc]

        if (tx < Br)
        {
#pragma unroll 32
            for (int i = 0; i < Bc; i++)
            {
                dSi[tx * Bc + i] = Pi[tx * Bc + i] * (dPi[tx * Bc + i] - Di_i[tx]);
            }
        }
        __syncthreads();

        mul_add_kA_B(dSi, Kj, dQi, Br, d, Bc, scale * GRAD_SCALE);  // dSi[Br x Bc] Kj[Bc x d]
        mul_add_kAT_B(dSi, Qi, dKj, Bc, d, Br, scale * GRAD_SCALE); // dSi[Br x Bc] Qi[Br x d]
    }
}

// =================================================================================

std::vector<torch::Tensor> forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    const int Br, const int Bc,
    const bool causal
)
{

    const int b = q.size(0);
    const int h = q.size(1);
    const int n = q.size(2);
    const int d = q.size(3);
    const int n_kv = k.size(2);

    const float32_t scale = 1.0 / sqrt(d);

    const int Tr = n / Br;
    const int Tc = n_kv / Bc;

    auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);
    auto O = torch::zeros_like(q, opt);

    auto opt2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto L = torch::zeros({b * h * n}, opt2);

    auto gridDim = dim3(b, h, Tr);
    auto blockDim = dim3(WAVE_SIZE * N_WAVES);

    const int sram_sz =
        Br * Bc * sizeof(ComputeType)  // Si
        + Br * d * sizeof(ComputeType) // Oi
        + Br * d * sizeof(ComputeType) // Qi
        ;

    cudaError_t err = cudaGetLastError();

    fwd_kernel<<<gridDim, blockDim, sram_sz>>>(
        (ComputeType *)q.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)k.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)v.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),
        (float *)L.data_ptr<float>(),
        Tr, Tc, Br, Bc, n, n_kv, d,
        scale, causal);

    err = cudaGetLastError();
    if (err != hipSuccess)
    {
        printf("=============== Kernel Launch Failed !!! =============\r\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Br:%d, Bc:%d \r\n", Br, Bc);
        printf("Tr:%d, Tc:%d \r\n", Tr, Tc);
        printf("B:%d, H:%d, Qn:%d, KVn:%d, d:%d \r\n", b, h, n, n_kv, d);
        printf("SRAM Requirements:%d \r\n", sram_sz);
    }

    return {O, L};
}

std::vector<torch::Tensor> backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L,
    const int Br,
    const int Bc,
    const bool causal
)
{
    const int b = Q.size(0);
    const int h = Q.size(1);
    const int n = Q.size(2);
    const int d = Q.size(3);

    const int n_kv = K.size(2);

    const float32_t scale = 1.0 / sqrt(d);

    const int Tr = n / Br;
    const int Tc = n_kv / Bc;

    auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);

    auto dQ = torch::zeros_like(Q, opt);
    auto dK = torch::zeros_like(K, opt);
    auto dV = torch::zeros_like(V, opt);

    auto optDi = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto Di = torch::zeros({b * h * n}, optDi);

    auto gridDim = dim3(b, h, Tc);
    auto blockDim = dim3(WAVE_SIZE * N_WAVES);

    const int sram_sz =
        2 * Br * Bc * sizeof(ComputeType) // (Si,Pi,dSi), dPi
        ;

    cudaError_t err = cudaGetLastError();

    bwd_kernel<<<gridDim, blockDim, sram_sz>>>(
        (ComputeType *)Q.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)K.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)V.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dO.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dQ.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dK.data_ptr<AT_PTR_TYPE>(),
        (ComputeType *)dV.data_ptr<AT_PTR_TYPE>(),
        (float *)Di.data_ptr<float>(),
        (float *)L.data_ptr<float>(),
        Tr, Tc,
        Br, Bc,
        n, n_kv, d,
        scale, causal);

    err = cudaGetLastError();
    if (err != hipSuccess)
    {
        printf("=============== Backward Kernel Launch Failed !!! =============\r\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Br:%d, Bc:%d \r\n", Br, Bc);
        printf("Tr:%d, Tc:%d \r\n", Tr, Tc);
        printf("B:%d, H:%d, Qn:%d, KVn:%d, d:%d \r\n", b, h, n, n_kv, d);
        printf("SRAM Requirements:%d \r\n", sram_sz);
    }

    return {dQ, dK, dV};
}
