#include "hash.cuh"

#define GGML_CUDA_HASH_BLOCK 256

static __device__ __forceinline__ uint64_t hash64_mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static __device__ __forceinline__ uint64_t hash64_word(uint64_t word, uint64_t idx) {
    return hash64_mix(word ^ ((idx + 1) * 0x9e3779b97f4a7c15ULL));
}

// grid-stride over the whole words; the first thread also folds in the zero-extended tail
static __global__ void hash_bytes_kernel(const uint64_t * words, size_t n_words, const uint8_t * tail, int n_tail,
                                         unsigned long long * acc_out) {
    __shared__ uint64_t s_acc[GGML_CUDA_HASH_BLOCK];

    uint64_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += (size_t) gridDim.x * blockDim.x) {
        acc ^= hash64_word(words[i], i);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_tail > 0) {
        uint64_t w = 0;
        for (int b = 0; b < n_tail; b++) {
            w |= (uint64_t) tail[b] << (8*b);
        }
        acc ^= hash64_word(w, n_words);
    }

    s_acc[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < (unsigned) stride) {
            s_acc[threadIdx.x] ^= s_acc[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicXor(acc_out, (unsigned long long) s_acc[0]);
    }
}

void ggml_cuda_hash_bytes(const void * data, size_t n, unsigned long long * d_acc, cudaStream_t stream) {
    const size_t n_words = n / 8;
    const int    n_tail  = (int) (n % 8);
    const size_t n_blocks_needed = (n_words + GGML_CUDA_HASH_BLOCK - 1) / GGML_CUDA_HASH_BLOCK;
    const int    n_blocks = (int) std::max<size_t>(1, std::min<size_t>(n_blocks_needed, 4096));
    hash_bytes_kernel<<<n_blocks, GGML_CUDA_HASH_BLOCK, 0, stream>>>(
        (const uint64_t *) data, n_words, (const uint8_t *) data + n_words*8, n_tail, d_acc);
}
