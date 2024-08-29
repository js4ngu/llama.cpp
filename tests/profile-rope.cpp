#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cstring>  // Add this line

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#define MAX_NARGS 3

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_SILU_FP16

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

static struct ggml_tensor * get_zero_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[]) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    // Fill the tensor with zeros
    memset(result->data, 0, ggml_nbytes(result));

    return result;
}

static struct ggml_tensor * get_sequential_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[]) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    // Initialize the tensor with sequential values starting from 1
    float * data = (float *) result->data;
    int64_t n_elements = ggml_nelements(result);
    for (int64_t i = 0; i < n_elements; ++i) {
        data[i] = (float)(i + 1);
    }
    return result;
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static void print_tensor_f32(const struct ggml_tensor * tensor) {
    float * data = (float *) tensor->data;
    int64_t n_elements = ggml_nelements(tensor);
    int a = 0;
    for (int64_t i = 0; i < n_elements; ++i) {
        printf("%f ", data[i]);
        if ((i + 1) % 10 == 0) {  // Print 10 elements per line
            printf("\t %d \n", a);
            a++;
        }
    }
    printf("\n");
}

static void print_tensor_i32(const struct ggml_tensor * tensor) {
    int32_t * data = (int32_t *) tensor->data;
    int64_t n_elements = ggml_nelements(tensor);
    for (int64_t i = 0; i < n_elements; ++i) {
        printf("%d ", data[i]);
        if ((i + 1) % 10 == 0) {  // Print 10 elements per line
            printf("\n");
        }
    }
    printf("\n");
}


int main(int /*argc*/, const char ** /*argv*/) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 128*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<uint8_t> work_buffer;
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_tensor * x;
    const int mode = 0;  // Set mode to 0 (standard RoPE)

    const int ndims = 4;
    const int64_t seqLen = 4096; 
    const int64_t n_rot = seqLen / 2;
    const int64_t numToken = 30; // 넘 작으면 안보임
    const int64_t numHead = 32; //변경하면 안됨
    const int64_t batch = 1; // 1로 해야 결과가 유리함 -> 커널 연산이 보임
    const int64_t ne[4] = { seqLen, numHead, numToken, batch };

    //const int ndims = 4;
    //const int64_t n_rot = 128;
    //const int64_t ne[4] = { 2*n_rot, 32, 73, 1 };

    struct ggml_tensor * p0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);

    for (int i = 0; i < ne[2]; ++i) {
        ((int32_t *) p0->data)[i] = 100+i;
    }

    //printf("Pos vec for RoPE:\n");
    //print_tensor_i32(p0);

    x = get_sequential_tensor_f32(ctx0, ndims, ne);
    //printf("before RoPE:\n");
    //print_tensor_f32(x);

    struct ggml_tensor * r0 = ggml_rope(ctx0, x,  p0, n_rot, mode);

    ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, r0);
    ggml_graph_compute_helper(work_buffer, gf, 2);
    //printf("after RoPE:\n");
    //print_tensor_f32(r0);
    ggml_free(ctx0);
    return 0;
}
