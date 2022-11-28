#include "easypap.h"
#include "global.h"
#include "img_data.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned int TYPE;
// typedef unsigned int unit;

static TYPE *restrict TABLE = NULL;
static TYPE *restrict UNSTABLE_TILES = NULL;

#define U_WIDTH DIM / TILE_W
#define U_HEIGHT DIM / TILE_H

static inline TYPE *atable_cell(TYPE *restrict i, int y, int x) {
    return i + y * DIM + x;
}

static inline TYPE *utable_cell(TYPE *restrict i, int y, int x) {
    return i + y * U_WIDTH + x;
}

#define atable(y, x) (*atable_cell(TABLE, (y), (x)))
#define utable(x, y) (*utable_cell(UNSTABLE_TILES, (y), (x)))
static inline void display_utable() {
    // display the array of unstable tiles
    for (int y = 0; y < U_HEIGHT; y++) {
        for (int x = 0; x < U_WIDTH; x++) {
            fprintf(stderr, "%d", utable(x, y));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

static inline void init_utable() {
    if (UNSTABLE_TILES == NULL) {
        UNSTABLE_TILES = calloc(U_WIDTH * U_HEIGHT, sizeof(TYPE));
        for (int i = 0; i < U_WIDTH * U_HEIGHT; i++) {
            UNSTABLE_TILES[i] = 1;
        }
    }
}

static inline TYPE *table_cell(TYPE *restrict i, int step, int y, int x) {
    return DIM * DIM * step + i + y * DIM + x;
}

#define table(step, y, x) (*table_cell(TABLE, (step), (y), (x)))

static int in = 0;
static int out = 1;

static inline void swap_tables() {
    int tmp = in;
    in = out;
    out = tmp;
}

static TYPE *restrict CACHE = NULL;

#define RGB(r, g, b) rgba(r, g, b, 0xFF)

static TYPE max_grains;

static inline void compute_cache(TYPE *restrict cache) {
    assert(CACHE != NULL);
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            cache[i * DIM + j] = table(in, i, j) / 4;
        }
}

#define init_cache (compute_cache(CACHE))
// Add inline function to retrieve index for (i,j)
static inline TYPE cache_value(TYPE *restrict cache, int x, int y) {
    assert(CACHE != NULL);
    return *(cache + y * DIM + x); // in 2D array
}
#define get_cache(y, x) (cache_value(CACHE, x, y))

void asandPile_refresh_img() {
    unsigned long int max = 0;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++) {
            int g = table(in, i, j);
            int r, v, b;
            r = v = b = 0;
            if (g == 1)
                v = 255;
            else if (g == 2)
                b = 255;
            else if (g == 3)
                r = 255;
            else if (g == 4)
                r = v = b = 255;
            else if (g > 4)
                r = b = 255 - (240 * ((double)g) / (double)max_grains);

            cur_img(i, j) = RGB(r, v, b);
            if (g > max)
                max = g;
        }
    max_grains = max;
}

/////////////////////////////  Initial Configurations

static inline void set_cell(int y, int x, unsigned v) {
    atable(y, x) = v;
    if (opencl_used)
        cur_img(y, x) = v;
}

void asandPile_draw_4partout(void);

void asandPile_draw(char *param) {
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper(param, asandPile_draw_4partout);
}

void ssandPile_draw(char *param) {
    hooks_draw_helper(param, asandPile_draw_4partout);
}

void asandPile_draw_4partout(void) {
    max_grains = 8;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
            set_cell(i, j, 4);
}

void asandPile_draw_DIM(void) {
    max_grains = DIM;
    for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
        for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
            set_cell(i, j, i * j / 4);
}

void asandPile_draw_alea(void) {
    max_grains = 5000;
    for (int i = 0; i < DIM >> 3; i++) {
        set_cell(1 + random() % (DIM - 2), 1 + random() % (DIM - 2),
                 1000 + (random() % (4000)));
    }
}

void asandPile_draw_big(void) {
    const int i = DIM / 2;
    set_cell(i, i, 100000);
}

static void one_spiral(int x, int y, int step, int turns) {
    int i = x, j = y, t;

    for (t = 1; t <= turns; t++) {
        for (; i < x + t * step; i++)
            set_cell(i, j, 3);
        for (; j < y + t * step + 1; j++)
            set_cell(i, j, 3);
        for (; i > x - t * step - 1; i--)
            set_cell(i, j, 3);
        for (; j > y - t * step - 1; j--)
            set_cell(i, j, 3);
    }
    set_cell(i, j, 4);

    for (int i = -2; i < 3; i++)
        for (int j = -2; j < 3; j++)
            set_cell(i + x, j + y, 3);
}

static void many_spirals(int xdebut, int xfin, int ydebut, int yfin, int step,
                         int turns) {
    int i, j;
    int size = turns * step + 2;

    for (i = xdebut + size; i < xfin - size; i += 2 * size)
        for (j = ydebut + size; j < yfin - size; j += 2 * size)
            one_spiral(i, j, step, turns);
}

static void spiral(unsigned twists) {
    many_spirals(1, DIM - 2, 1, DIM - 2, 2, twists);
}

void asandPile_draw_spirals(void) { spiral(DIM / 32); }

// shared functions

#define ALIAS(fun)                                                             \
    void ssandPile_##fun() { asandPile_##fun(); }

ALIAS(refresh_img);
ALIAS(draw_4partout);
ALIAS(draw_DIM);
ALIAS(draw_alea);
ALIAS(draw_big);
ALIAS(draw_spirals);

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Synchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void ssandPile_init() {
    size_t size = 3 * DIM * DIM * sizeof(TYPE);
    TABLE = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CACHE = calloc(DIM * DIM, sizeof(TYPE));
    init_utable();
}

void ssandPile_finalize() { munmap(TABLE, 3 * DIM * DIM * sizeof(TYPE)); }

int ssandPile_do_tile_default(int x, int y, int width, int height) {
    int diff = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            table(out, i, j) = table(in, i, j) % 4;
            table(out, i, j) += table(in, i + 1, j) / 4;
            table(out, i, j) += table(in, i - 1, j) / 4;
            table(out, i, j) += table(in, i, j + 1) / 4;
            table(out, i, j) += table(in, i, j - 1) / 4;
            if (table(out, i, j) >= 4)
                diff = 1;
        }

    return diff;
}

#pragma GCC push_options
#pragma GCC optimize("unroll-all-loops")
/**
 * 4.1: Make use of auto-vectorization to optimize the
 * ssandPile_do_tile_default() function.
 *
 * usage
 * ./run -k ssandPile -s 256 -wt opt -m
 */
int ssandPile_do_tile_opt(int x, int y, int width, int height) {
    int diff = 0;
    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            table(out, i, j) = table(in, i, j) % 4 + table(in, i + 1, j) / 4 +
                               table(in, i - 1, j) / 4 +
                               table(in, i, j + 1) / 4 +
                               table(in, i, j - 1) / 4;
            diff |= table(out, i, j) / 4;
        }
    return diff;
}

/**
 * 4.1: Make use of auto-vectorization to optimise the
 * ssandPile_do_tile_default() function. A cache has been added to to cache
 * where we cache the value of each cell /4 since it's always calculated. It is
 * not faster actually a bit slower..
 *
 * usage:
 * ./run -k ssandPile -s 256 -wt opt1 -m
 *
 */
int ssandPile_do_tile_opt1(int x, int y, int width, int height) {
    int diff = 0;
    init_cache;
    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            table(out, i, j) = table(in, i, j) % 4 + get_cache(i + 1, j) +
                               get_cache(i - 1, j) + get_cache(i, j + 1) +
                               get_cache(i, j - 1);
            diff |= table(out, i, j) / 4;
        }
    return diff;
}

/**
 * 4.2: Function that uses thread to compute the table
 *
 * usage:
 * ./run -k ssandPile -s 256 -v omp -m
 */
static inline int ssandPile_do_tile_omp(int x, int y, int width, int height) {
    int diff = 0;
#pragma omp parallel for schedule(runtime) reduction(| : diff)
    for (int i = y; i < y + height; i++) {
        monitoring_start_tile(omp_get_thread_num());
        for (int j = x; j < x + width; j++) {
            table(out, i, j) = table(in, i, j) % 4 + table(in, i + 1, j) / 4 +
                               table(in, i - 1, j) / 4 +
                               table(in, i, j + 1) / 4 +
                               table(in, i, j - 1) / 4;
            if (table(out, i, j) >= 4)
                diff = 1;
        }
        monitoring_end_tile(x, y, width, height, omp_get_thread_num());
    }
    return diff;
}

static inline int is_32_byte_aligned(TYPE *adress) {
    return ((size_t)adress & 31);
}

static inline void check32_alignment(int i, int j) {
    int res = is_32_byte_aligned(&table(in, i, j));
    if (res)
        fprintf(stderr,
                "res: %d, adress for x: %d, y: %d, is not 32 byte aligned\n",
                res, j, i);
}

static void log_256i(__m256i vec, char *name) {
    fprintf(stderr, "%s: [", name);
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 0));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 1));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 2));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 3));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 4));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 5));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 6));
    fprintf(stderr, " %d, ", _mm256_extract_epi32(vec, 7));
    fprintf(stderr, "]\n");
}

static void log_tile_info(int x, int y, int width, int height) {
    fprintf(stderr, "x: %d, y: %d, width: %d, height: %d\n", x, y, width,
            height);
}

static inline void _mm256_store(void *dest, __m256i src) {
    *((int *)dest + 0) = _mm256_extract_epi32(src, 0);
    *((int *)dest + 1) = _mm256_extract_epi32(src, 1);
    *((int *)dest + 2) = _mm256_extract_epi32(src, 2);
    *((int *)dest + 3) = _mm256_extract_epi32(src, 3);
    *((int *)dest + 4) = _mm256_extract_epi32(src, 4);
    *((int *)dest + 5) = _mm256_extract_epi32(src, 5);
    *((int *)dest + 6) = _mm256_extract_epi32(src, 6);
    *((int *)dest + 7) = _mm256_extract_epi32(src, 7);
}
/**
 * 4.5.1: Make use of explicit-vectorization to optimize the
 * ssandPile_do_tile_default() function. This version treats border
 * tiles differenty and calls for those functions
 * ssandPile_do_tile_opt()
 *
 * usage
 * ./run -k ssandPile -s 256 -wt avx -m
 */
int ssandPile_do_tile_avx(int x, int y, int width, int height) {
    if (x <= 1 || y <= 1 || x >= DIM - width - 1 || y >= DIM - height - 1)
        return ssandPile_do_tile_opt(x, y, width, height);

    int diff = 0;
    __m256i input, input_top, input_bottom, input_left, input_right, output,
        three;
    three = _mm256_set1_epi32(3);
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += 8) {

            // load in data in vectors
            input = _mm256_loadu_si256((__m256i *)&table(in, i, j));
            input_top = _mm256_loadu_si256((__m256i *)&table(in, i - 1, j));
            input_bottom = _mm256_loadu_si256((__m256i *)&table(in, i + 1, j));
            input_left = _mm256_loadu_si256((__m256i *)&table(in, i, j - 1));
            input_right = _mm256_loadu_si256((__m256i *)&table(in, i, j + 1));

            // mod 4 => table(in, i, j) % 4
            output = _mm256_and_si256(input, three);

            output += _mm256_add_epi32(_mm256_srli_epi32(input_top, 2),
                                       _mm256_srli_epi32(input_bottom, 2)) +
                      _mm256_add_epi32(_mm256_srli_epi32(input_left, 2),
                                       _mm256_srli_epi32(input_right, 2));

            _mm256_store(&table(out, i, j), output);

            diff |= _mm256_movemask_epi8(_mm256_cmpgt_epi32(output, three));
        }
    }
    return diff;
}

/**
 * 4.5.1: Make use of explicit-vectorization to optimize the
 * ssandPile_do_tile_default() function. This version
 * looks at all tiles and uses AVX256
 *
 * usage
 * ./run -k ssandPile -s 256 -wt avx2 -m
 */
int ssandPile_do_tile_avx2(int x, int y, int width, int height) {
    int diff = 0;
    x = x * (x != 1);
    width = width + (width < TILE_W);
    height = height + (height < TILE_W);
    __m256i input, input_top, input_bottom, input_left, input_right, output,
        three;
    three = _mm256_set1_epi32(3);
    for (int i = y; i < y + height && i < DIM - 1; i++) {
        for (int j = x; j < x + width; j += 8) {
            int left_neighbor_exist = j > 1;
            int right_neighbor_exist = (j + 7) < (DIM - 1);

            // load data in vectors
            input = _mm256_load_si256((__m256i *)&table(in, i, j));
            input_left = _mm256_loadu_si256((__m256i *)&table(in, i, j - 1));
            input_right = _mm256_loadu_si256((__m256i *)&table(in, i, j + 1));

#ifdef __AVX512F__
            input_left =
                _mm256_maskz_add_epi32(0xff >> !right_neighbor_exist,
                                       _mm256_setzero_si256(), input_left);
            input_right =
                _mm256_maskz_add_epi32(0xff << !left_neighbor_exist,
                                       _mm256_setzero_si256(), input_right);
#else
            // if rightest neighbor is a border cell, then his left cell should
            // be 0
            input_left = _mm256_insert_epi32(
                input_left, table(in, i, j + 7) * right_neighbor_exist, 7);

            // if leftest neighbor is a border cell, then his right cell should
            // be 0
            input_right = _mm256_insert_epi32(
                input_right, table(in, i, j + 1) * left_neighbor_exist, 0);
#endif /* __AVX512F__ */

            input_top = _mm256_loadu_si256((__m256i *)&table(in, i - 1, j));
            input_bottom = _mm256_loadu_si256((__m256i *)&table(in, i + 1, j));

            output = _mm256_add_epi32(
                _mm256_and_si256(input, three),                // input % 4
                _mm256_add_epi32(                              //
                    _mm256_add_epi32(                          // +
                        _mm256_srli_epi32(input_left, 2),      // left/4 +
                        _mm256_srli_epi32(input_right, 2)),    // right/4
                    _mm256_add_epi32(                          // +
                        _mm256_srli_epi32(input_top, 2),       // top/4 +
                        _mm256_srli_epi32(input_bottom, 2)))); // bottom/4

            _mm256_store(&table(out, i, j), output);
            diff |= _mm256_movemask_epi8(_mm256_cmpgt_epi32(output, three));
        }
    }
    return diff;
}

#ifdef __AVX512F__

static void log_512i(__m512i vec) {
    int tab[16];
    fprintf(stderr, "[ ");
    _mm512_store_epi32(&tab, vec);
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, "%d, ", tab[i]);
    }
    fprintf(stderr, "]\n");
}

/**
 * 4.5.1: Make use of explicit-vectorization to optimize the
 * ssandPile_do_tile_default() function. This version
 * looks at all tiles and uses AVX512
 *
 * usage
 * ./run -k ssandPile -s 256 -wt avx5 -m
 */
int ssandPile_do_tile_avx5(int x, int y, int width, int height) {
    int diff = 0;

    x = x * (x != 1); // set back x to 0 if 1
    width = width + (width < TILE_W);
    height = height + (height < TILE_W);

    __m512i input, input_top, input_bottom, input_left, input_right, output,
        three, zero;

    three = _mm512_set1_epi32(3);
    zero = _mm512_setzero_epi32();
    for (int i = y; i < y + height && i < DIM - 1; i++) {
        for (int j = x; j < x + width; j += 16) {
            int left_neighbor_exist = j > 1;
            int right_neighbor_exist = (j + 15) < (DIM - 1);

            // load in data in vectors
            input = _mm512_load_epi32(&table(in, i, j));

            input_left = _mm512_loadu_epi32(&table(in, i, j - 1));
            input_left = _mm512_maskz_add_epi32(0xffff >> !right_neighbor_exist,
                                                zero, input_left);

            input_right = _mm512_loadu_epi32(&table(in, i, j + 1));
            input_right = _mm512_maskz_add_epi32(0xffff << !left_neighbor_exist,
                                                 zero, input_right);

            input_top = _mm512_load_epi32(&table(in, i - 1, j));
            input_bottom = _mm512_load_epi32(&table(in, i + 1, j));

            // mod 4 => table(in, i, j) % 4
            output = _mm512_add_epi32(
                _mm512_and_epi32(input, three),                // input % 4
                _mm512_add_epi32(                              //
                    _mm512_add_epi32(                          // +
                        _mm512_srli_epi32(input_left, 2),      // left/4 +
                        _mm512_srli_epi32(input_right, 2)),    // right/4
                    _mm512_add_epi32(                          // +
                        _mm512_srli_epi32(input_top, 2),       // top/4 +
                        _mm512_srli_epi32(input_bottom, 2)))); // bottom/4

            _mm512_store_epi32(&table(out, i, j), output);
            diff |= _mm512_cmpgt_epi32_mask(output, three);
        }
    }
    return diff;
}

#endif /* __AVX512F__ */

#pragma GCC pop_options

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned ssandPile_compute_seq(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = do_tile(1, 1, DIM - 2, DIM - 2, 0);
        swap_tables();
        if (change == 0)
            return it;
    }
    return 0;
}

/**
 * 4.2: An omp implementation without using tiles
 *
 * usage:
 * ./run -k ssandPile -s 256 -v omp -m
 */
unsigned ssandPile_compute_omp(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = ssandPile_do_tile_omp(1, 1, DIM - 2, DIM - 2);
        swap_tables();
        if (change == 0)
            return it;
    }
    return 0;
}

unsigned ssandPile_compute_tiled(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  0 /* CPU id */);
        swap_tables();
        if (change == 0)
            return it;
    }

    return 0;
}

void ssandPile_init_ocl(void){
    fprintf(stderr, "Initialize normal ocl\n");
    ssandPile_init();
}

unsigned ssandPile_invoke_ocl(unsigned nb_iter) {
    size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
    size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
    cl_int err;

    monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

    for (unsigned it = 1; it <= nb_iter; it++) {
        // Set kernel arguments
        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        check (err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);
        check (err, "Failed to execute kernel");

        // Swap buffers
        {
            cl_mem tmp  = cur_buffer;
            cur_buffer  = next_buffer;
            next_buffer = tmp;
        }
    }
    clFinish (queue);
    monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
    return 0;
}

static void gpu_info(){
    fprintf(stderr, "{GPU_SIZE_X: %d, GPU_SIZE_Y: %d}, {GPU_TILE_W: %d, GPU_TILE_H: %d}\n",
    GPU_SIZE_X, GPU_SIZE_Y, GPU_TILE_W, GPU_TILE_H);
}

// #define CHECK_ITER (unsigned) (15944040000.000002 + (-0.5441822 - 15944040000.000002)/(1 + pow((DIM/230808.0),2.020573))) + 10
#define EXPECTED_ITERATION (unsigned) (1.49e-07 * pow(DIM, 4) - 0.0001171 * pow(DIM, 3) + 0.2898 * pow(DIM, 2) - 2.583 * DIM + 30.62) 
static cl_mem buffer;
static unsigned iterations;
static unsigned expected_iteration;


void ssandPile_init_ocl_term(void){
    fprintf(stderr, "Initialize term ocl\n");
    ssandPile_init();
    // iterations = 0;
    // expected_iteration = 1.49e-07 * pow(DIM, 4) - 0.0001171 * pow(DIM, 3) + 0.2898 * pow(DIM, 2) - 2.583 * DIM + 30.62;
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned) * DIM * DIM, NULL, NULL);
    // gpu_info();
}

static inline int is_stable(unsigned *table){
    for(int i = 0; i < DIM*DIM; i++){
        if(table[i])
            return 0;
    }
    return 1;
}


static int inline check_stability(unsigned *table){
    if(expected_iteration == iterations){
        iterations = 0;
        fprintf(stderr, "IS ZERO\n");
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, table, 0, NULL, NULL);
        if(is_stable(table)){
            return 1;
        }
    }
    return 0;
}

unsigned ssandPile_invoke_ocl_term (unsigned nb_iter) {
    size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
    size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
    cl_int err;
    unsigned table[DIM*DIM];

    monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

    for (unsigned it = 1; it <= nb_iter; it++, iterations++) {
        // Set kernel arguments
        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &buffer);
        check (err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);
        check (err, "Failed to execute kernel");

        // Swap buffers
        {
            cl_mem tmp  = cur_buffer;
            cur_buffer  = next_buffer;
            next_buffer = tmp;
        }
        // if(check_stability(table)){
        //     clFinish (queue);
        //     monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
        //     return it;
        // }
    }
    clFinish (queue);
    monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
    return 0;
}

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl() {
    cl_int err;

    err =
        clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                            sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    ssandPile_refresh_img();
}

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl_term() {
    cl_int err;

    err =
        clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                            sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    ssandPile_refresh_img();
}


#define THRESHOLD 10
#define NB_LINES_TO_COPY 10
static unsigned cpu_y_begin; // the cpu does the tile from 0 to cpu_y_end
static unsigned gpu_y_end; // the gpu does the tile from gpu_y_begin to DIM
static unsigned valid_copied_lines;
static long gpu_duration = 0;
static long cpu_duration = 0;

static void debug(size_t *global){
    fprintf(stderr, "GPU_SIZE_X: %ld, GPU_SIZE_Y: %ld, cpu_begin: %d, gpu_end: %d\n",
        global[0], global[1], cpu_y_begin, gpu_y_end);
}


void ssandPile_init_ocl_hybrid(void){
    fprintf(stderr, "Initialization\n");
    ssandPile_init();
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned) * DIM * DIM, NULL, NULL);
    // cpu_y_part = (NB_TILES_Y/2) * GPU_TILE_H;
    gpu_y_end = (NB_TILES_Y/2) * GPU_TILE_H; //cautious GPU_TILE_H is not always same as TILE_H
    cpu_y_begin = gpu_y_end;
    valid_copied_lines = 0;
    fprintf(stderr, "GPU_TILE_W: %d, GPU_TILE_H: %d\n", GPU_TILE_W, GPU_TILE_H);
    fprintf(stderr, "gpu end: %d, cpu begin : %d \n", gpu_y_end, cpu_y_begin);
    fprintf(stderr, "\n");
}

// return true if the difference t1 t2 is bigger 
static int compare_time(long t1, long t2, long threshold){
    return (t1 > t2) && ((t1-t2)*100/t1 > threshold);
}

static inline void balance_load(size_t *global){
    if(gpu_y_end < DIM - NB_LINES_TO_COPY - ( GPU_TILE_H) 
        && compare_time(cpu_duration, gpu_duration, THRESHOLD)){
        // copy the missing part from cpu to gpu
        check(clEnqueueWriteBuffer(queue, cur_buffer, CL_TRUE, 
                            sizeof(unsigned) * DIM * (cpu_y_begin),
                            sizeof(unsigned) * DIM * GPU_TILE_H, 
                            table_cell(TABLE, in, cpu_y_begin, 0), 
                            0, NULL, NULL),
                            "Failed to Write to queue");
        // fprintf(stderr, "changing cpu/gpu border\n");
        global[1] += GPU_TILE_H;
        gpu_y_end += GPU_TILE_H;
        cpu_y_begin = gpu_y_end;
        // debug(global);
        // fprintf(stderr, "\n");
    }
}

static inline void share_data_cpu_gpu(){
    // gpu to cpu
    check(clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 
                        sizeof(unsigned) * DIM * (gpu_y_end-NB_LINES_TO_COPY), 
                        sizeof(unsigned) * DIM * NB_LINES_TO_COPY, 
                        table_cell(TABLE, in, gpu_y_end-NB_LINES_TO_COPY, 0), 
                        0, NULL, NULL),
                        "Failed to Read from queue");
    // cpu to gpu
    check(clEnqueueWriteBuffer(queue, cur_buffer, CL_TRUE, 
                        sizeof(unsigned) * DIM * (cpu_y_begin),
                        sizeof(unsigned) * DIM * NB_LINES_TO_COPY, 
                        table_cell(TABLE, in, cpu_y_begin, 0), 
                        0, NULL, NULL), "Failed to Write to queue");
}

unsigned ssandPile_invoke_ocl_hybrid (unsigned nb_iter) {
    size_t global[2] = {GPU_SIZE_X, gpu_y_end + GPU_TILE_H}; // global domain size for our calculation
    size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
    cl_int err;
    long t1, t2;
    cl_event kernel_event;
    monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));
    for (unsigned it = 1; it <= nb_iter; it++, iterations++, valid_copied_lines--) {
        /*-------------------------------------------------------------------------------------------*/
        /*------------------ Share data and load between cpu and gpu --------------------------------*/
        /*-------------------------------------------------------------------------------------------*/
        if(valid_copied_lines <= 1){
            valid_copied_lines = NB_LINES_TO_COPY;
            balance_load(global);
            share_data_cpu_gpu();
        }
        /*-------------------------------------------------------------------------------------------*/
        /*--------------------------------------- GPU PART ------------------------------------------*/
        /*-------------------------------------------------------------------------------------------*/
        // Set kernel arguments
        // fprintf(stderr, "gpu end before setting kernel args: %d\n", gpu_y_end);
        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &buffer);
        err |= clSetKernelArg(compute_kernel, 3, sizeof(unsigned), &gpu_y_end);
        err |= clSetKernelArg(compute_kernel, 4, sizeof(unsigned), &valid_copied_lines);
        check (err, "Failed to set kernel arguments");

        // Launch GPU kernel
        err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, 
                                    global, local, 0, NULL, &kernel_event);
        check (err, "Failed to execute kernel");

        // Swap buffers
        {
            cl_mem tmp  = cur_buffer;
            cur_buffer  = next_buffer;
            next_buffer = tmp;
        }
        /*-------------------------------------------------------------------------------------------*/
        /*--------------------------------------- CPU PART ------------------------------------------*/
        /*-------------------------------------------------------------------------------------------*/
        t1 = what_time_is_it();
        #pragma omp parallel for collapse(2) schedule(runtime)
        for(int y = cpu_y_begin; y < DIM; y+=TILE_H){
            for(int x = 0; x < DIM; x += TILE_W){
                int begin_x = x + (x == 0);
                int begin_y = y + (y == 0) - ((y == cpu_y_begin)*(valid_copied_lines-1));
                int width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
                int height = TILE_H - ((y + TILE_H == DIM) + (y == 0)) 
                            + ((y == cpu_y_begin)*(valid_copied_lines-1));
                ssandPile_do_tile_opt(begin_x, begin_y, width, height);
            }
        }
        swap_tables();
        gpu_duration = ocl_monitor(kernel_event, 0, gpu_y_end, global[0],
                                global[1], TASK_TYPE_COMPUTE);

        // Measure time
        t2 = what_time_is_it();
        cpu_duration = t2 - t1;
        // debug_gpu(global);
    }
    clFinish (queue);
    clReleaseEvent(kernel_event);
    monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
    return 0;
}

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl_hybrid() {
    cl_int err;

    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                            sizeof(unsigned) * DIM * gpu_y_end, table_cell(TABLE, in, 0, 0), 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    ssandPile_refresh_img();
}


/**
 * 4.2: An OMP implementation using tiles and the collapse clause
 *
 * usage:
 * ./run -k ssandPile -s 256 -v omp_tiled -wt opt -m
 */
unsigned ssandPile_compute_omp_tiled(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel for schedule(runtime) reduction(| : change) collapse(2)
        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  omp_get_thread_num());
        swap_tables();
        if (change == 0)
            return it;
    }

    return 0;
}
/**
 * 4.2: An OMP implementation using task loops
 *
 * usage:
 * ./run -k ssandPile -s 256 -v omp_tasklopp -wt opt -m
 */
unsigned ssandPile_compute_omp_taskloop(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel shared(change) shared(it)
#pragma omp single
        {
            for (int y = 0; y < DIM; y += TILE_H)
#pragma omp task
                for (int x = 0; x < DIM; x += TILE_W)
                    change |= do_tile(x + (x == 0), y + (y == 0),
                                      TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                      TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                      omp_get_thread_num());
        }
        swap_tables();
        if (change == 0)
            return it;
    }

    return 0;
}

static inline void copy_in_tile_out(int x, int y, int tile_width,
                                    int tile_height) {
    for (int i = y; i < y + tile_height; i++) {
        for (int j = x; j < x + tile_width; j++) {
            table(out, i, j) = table(in, i, j);
        }
    }
}
/**
 * This function checks if a tile is unstable.
 * A tile is unstable if there are some cells in the tile that contains 4 or
 * more particles of sand.
 *
 * If the tile is unstable it is computed and the function updates the state
 * of the neigbouring tiles
 *
 * let's look at an example
 *
 * EXAMPLE :
 *
 * --- Before Computation ---
 *
 * `Stable tile 3x3 (A)`               `Unstable tile 3x3 (B) `
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 0 | 0 | 3 |
 *     -------------   in contact with    -------------
 *     | 0 | 0 | 0 | ---------------------| 0 | 16| 0 |
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 0 | 1 | 2 |
 *     -------------                      -------------
 *
 * --- After Computation (Synchronous) ---
 *
 * `Stable tile 3x3 (A)`               `Unstable tile 3x3 (B) `
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 0 | 4 | 0 |
 *     -------------   in contact with    -------------
 *     | 0 | 0 | 0 | ---------------------| 4 | 0 | 4 |
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 0 | 5 | 2 |
 *     -------------                      -------------
 *
 * In this example we can see that after computation in the (B) tile we have
 * a cell with 4 grain of sand in direct contact with an unstable cell If
 * the (A) tile stays unstable it won't updated
 *
 * So after each computation we check for the border of each tile if the
 * value in each cell is >= 4. If so all the neighbouring tile in contact
 * with that cell will become unstable
 *  */
static inline int sync_lazy_do_tile(int x, int y) {
    int change = 0;
    int ux = x / TILE_W;
    int uy = y / TILE_H;
    if (utable(ux, uy)) { // check if tile is stable
        int startx = x + (x == 0);
        int starty = y + (y == 0);
        int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
        int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));

        change |= do_tile(startx, starty, tile_width, tile_height,
                          omp_get_thread_num());
        if (change == 0)
            utable(ux, uy)--;
        else {
            utable(ux, uy) = 2;
        }

        // look if the top neigbour should be unstable
        if (uy > 0) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (table(out, starty, i) >= 4) {
                    utable(ux, uy - 1) = 3;
                    // change |= utable(ux, uy - 1);
                }
            }
        }
        // look if the bottom neigbour should be unstable
        if (uy < U_HEIGHT - 1) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (table(out, starty + tile_height - 1, i) >= 4) {
                    utable(ux, uy + 1) = 3;
                    // change |= utable(ux, uy + 1);
                }
            }
        }
        // look if the left neigbour should be unstable
        if (ux > 0) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (table(out, i, startx) >= 4) {
                    utable(ux - 1, uy) = 3;
                    change |= utable(ux - 1, uy);
                }
            }
        }

        // look if the right neigbour should be unstable
        if (ux < U_WIDTH - 1) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (table(out, i, startx + tile_width - 1) >= 4) {
                    utable(ux + 1, uy) = 3;
                    // change |= utable(ux + 1, uy);
                }
            }
        }
    }
    return utable(ux, uy) > 0;
}

static void display_tile(int x, int y, int width, int height) {
    fprintf(stderr, "Tile x:%d, y:%d\n", x, y);
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            fprintf(stderr, "%d", table(in, i, j));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "-------------------------------------\n");
}

static void display_out_tile(int x, int y, int width, int height) {
    fprintf(stderr, "Tile x:%d, y:%d\n", x, y);
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            fprintf(stderr, "%d", table(out, i, j));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "-------------------------------------\n");
}
static unsigned int is_tile_unstable(int x, int y, int width, int height) {
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            if (table(in, i, j) >= 4)
                return 1;
        }
    }
    return 0;
}

static unsigned int check_unstable_tiles() {
    unsigned int count = 0;
    for (int y = 0; y < DIM; y += TILE_H) {
        for (int x = 0; x < DIM; x += TILE_W) {
            int startx = x + (x == 0);
            int starty = y + (y == 0);
            int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
            int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));
            if (is_tile_unstable(startx, starty, tile_width, tile_height)) {
                display_tile(startx, starty, tile_width, tile_height);
                count++;
            }
        }
    }
    return count;
}
static void debug_tiles() {
    int count = check_unstable_tiles();
    if (count) {
        fprintf(stderr, "Error %d tiles are still unstable !\n", count);
    }
}

/**
 * 4.4: Lazy implementation synchronous.
 * To be able to have cpu to work only where it needs
 * the tiles are marked as unstable or unstable with the utable(),
 *
 * [see UNSTABLE_TILES]
 *
 * The call do_tile is therefore only done on unstable tiles
 *
 * usage:
 * ./run -k ssandPile -s 512 -wt opt -v -a alea lazy -m
 *
 */
unsigned ssandPile_compute_lazy(unsigned nb_iter) {
    int change = 0;
    for (unsigned it = 1; it <= nb_iter; it++) {
        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                change |= sync_lazy_do_tile(x, y);
            }
        }
        swap_tables();
        if (change == 0) {
            return it;
        }
    }

    return 0;
}

static inline int sync_lazy_do_tile_omp(int x, int y) {
    int change = 0;
    int ux = x / TILE_W;
    int uy = y / TILE_H;
    if (utable(ux, uy)) { // check if tile is stable
        int startx = x + (x == 0);
        int starty = y + (y == 0);
        int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
        int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));

        change |= do_tile(startx, starty, tile_width, tile_height,
                          omp_get_thread_num());
        if (change == 0) {
#pragma omp atomic
            utable(ux, uy)--;
        } else {
            utable(ux, uy) = 2;
        }

        // utable(ux, uy) = 0;
        // utable(ux, uy) |= change;

        // look if the top neigbour should be unstable
        if (uy > 0) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (table(out, starty, i) >= 4) {
                    utable(ux, uy - 1) = 3;
                    // change |= utable(ux, uy - 1);
                }
            }
        }
        // look if the bottom neigbour should be unstable
        if (uy < U_HEIGHT - 1) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (table(out, starty + tile_height - 1, i) >= 4) {
                    utable(ux, uy + 1) = 3;
                    // change |= utable(ux, uy + 1);
                }
            }
        }
        // look if the left neigbour should be unstable
        if (ux > 0) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (table(out, i, startx) >= 4) {
                    utable(ux - 1, uy) = 3;
                    change |= utable(ux - 1, uy);
                }
            }
        }

        // look if the right neigbour should be unstable
        if (ux < U_WIDTH - 1) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (table(out, i, startx + tile_width - 1) >= 4) {
                    utable(ux + 1, uy) = 3;
                    // change |= utable(ux + 1, uy);
                }
            }
        }
    }
    return utable(ux, uy) > 0;
}
/**
 * 4.4: Lazy OMP implementation synchronous.
 * To be able to have cpu to work only where it needs
 * the tiles are marked as unstable or unstable with the utable(),
 *
 * [see UNSTABLE_TILES]
 * The call do_tile is therefore only done on unstable tiles
 *
 * usage:
 * ./run -k ssandPile -s 512 -wt opt -v -a alea omp_lazy -m
 *
 */
unsigned ssandPile_compute_omp_lazy(unsigned nb_iter) {
    int change = 0;
    for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel for schedule(runtime) reduction(| : change) collapse(2)
        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
                change |= sync_lazy_do_tile_omp(x, y);

        if (change == 0) {
            return it;
        } else {
            swap_tables();
        }
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Asynchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void asandPile_init() {
    in = out = 0;
    if (TABLE == NULL) {
        const unsigned size = DIM * DIM * sizeof(TYPE);

        PRINT_DEBUG('u', "Memory footprint = 2 x %d bytes\n", size);

        TABLE = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    init_utable();
}

void asandPile_finalize() {
    const unsigned size = DIM * DIM * sizeof(TYPE);

    munmap(TABLE, size);
}

///////////////////////////// Version séquentielle simple (seq)
// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

int asandPile_do_tile_default(int x, int y, int width, int height) {
    int change = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
            if (atable(i, j) >= 4) {
                atable(i, j - 1) += atable(i, j) / 4;
                atable(i, j + 1) += atable(i, j) / 4;
                atable(i - 1, j) += atable(i, j) / 4;
                atable(i + 1, j) += atable(i, j) / 4;
                atable(i, j) %= 4;
                change = 1;
            }
    return change;
}

#pragma GCC push_options
#pragma GCC optimize("unroll-all-loops")
/**
 * 4.1: Make use of auto-vectorization to optimise the
 * ssandPile_do_tile_default() function. A cache has been added to to cache
 * the value atable(i,  j)/4 It does not bring that much of an improvement
 * but it's usually a tiny bit faster.
 *
 * usage:
 * ./run -k asandPile -s 256 -wt opt -m
 */
int asandPile_do_tile_opt(int x, int y, int width, int height) {
    int change = 0;
    TYPE cache = 0;
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            cache = atable(i, j) / 4;
            if (!cache)
                continue;
            atable(i, j - 1) += cache;
            atable(i, j + 1) += cache;
            atable(i - 1, j) += cache;
            atable(i + 1, j) += cache;
            atable(i, j) %= 4;
            change |= cache;
        }
    }
    return change;
}

/**
 * 4.1: Make use of auto-vectorization to optimise the
 * ssandPile_do_tile_default() function. Here we didn't cache any value
 *
 * usage:
 * ./run -k asandPile -s 256 -wt opt2 -m
 */
int asandPile_do_tile_opt1(int x, int y, int width, int height) {
    int change = 0;
    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            if (atable(i, j) >= 4) {
                atable(i, j - 1) += atable(i, j) / 4;
                atable(i, j + 1) += atable(i, j) / 4;
                atable(i - 1, j) += atable(i, j) / 4;
                atable(i + 1, j) += atable(i, j) / 4;
                atable(i, j) %= 4;
                change = 1;
            }
        }
    return change;
}

static inline void _mm256_store_avx(void *dest, __m256i src, int left,
                                    int right) {
    *((int *)dest + 1 - left) = _mm256_extract_epi32(src, 0);
    *((int *)dest + 1) = _mm256_extract_epi32(src, 1);
    *((int *)dest + 2) = _mm256_extract_epi32(src, 2);
    *((int *)dest + 3) = _mm256_extract_epi32(src, 3);
    *((int *)dest + 4) = _mm256_extract_epi32(src, 4);
    *((int *)dest + 5) = _mm256_extract_epi32(src, 5);
    *((int *)dest + 6 + right) = _mm256_extract_epi32(src, 7);
    *((int *)dest + 6) = _mm256_extract_epi32(src, 6);
}

#ifdef __AVX512F__
/**
 * 4.5.2: Make use of explicit-vectorization to optimize the
 * asandPile_do_tile_default() function. This version
 * looks at all tiles and uses AVX256
 *
 * usage
 * ./run -k asandPile -s 256 -wt avx2 -m
 */
int asandPile_do_tile_avx2(int x, int y, int width, int height) {
    // all tiles are calculated with avx
    //  assumption DIM is divisible by 8
    int diff = 0;
    x = x * (x != 1);
    width =
        width + (x + TILE_W == DIM) +
        (x == 0); // if tile is not divisible by 8 we include the outer border
    height = height + (y + TILE_H == DIM) + (y == 0);

    __m256i input, input_top, input_bottom, three, DIV, shift_left, shift_right;
    three = _mm256_set1_epi32(3);

    for (int i = y; i < y + height && i < DIM - 1; i++) {
        for (int j = x; j < x + width; j += 8) {
            // we check if the left and right neighbor exists and if they should
            // be zero or not
            int left_neighbor_non_zero = j > 1;
            int right_neighbor_non_zero = j + 7 < (DIM - 1);
            // load data in vectors, if neighbor line doesn't exist, we load the
            // same line

            input = _mm256_loadu_si256((__m256i *)&atable(i, j));
            input_top = _mm256_loadu_si256((__m256i *)&atable(i - 1, j));
            input_bottom = _mm256_loadu_si256((__m256i *)&atable(i + 1, j));

            // DIV 4
            DIV = _mm256_srli_epi32(input, 2);

            shift_left = _mm256_alignr_epi32(_mm256_setzero_si256(), DIV, 1);
            shift_right = _mm256_alignr_epi32(DIV, _mm256_setzero_si256(), 7);

            // add and modulo
            input = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_and_si256(input, three), shift_left),
                shift_right);

            input_top = _mm256_add_epi32(
                input_top, _mm256_maskz_add_epi32(0xff * (i > 1), DIV,
                                                  _mm256_setzero_si256()));
            input_bottom = _mm256_add_epi32(
                input_bottom, _mm256_maskz_add_epi32(0xff * (i < DIM - 2), DIV,
                                                     _mm256_setzero_si256()));

            // store if neighbors are existant
            atable(i, j - left_neighbor_non_zero) +=
                _mm256_extract_epi32(DIV, 0);

            atable(i, j + 7 + right_neighbor_non_zero) +=
                _mm256_extract_epi32(DIV, 7);

            _mm256_store_avx(&atable(i - 1, j), input_top,
                             left_neighbor_non_zero, right_neighbor_non_zero);
            _mm256_store_avx(&atable(i + 1, j), input_bottom,
                             left_neighbor_non_zero, right_neighbor_non_zero);
            _mm256_store_avx(&atable(i, j), input, left_neighbor_non_zero,
                             right_neighbor_non_zero);

            // check if one of them is still above 3
            diff |=
                (atable(i, j - left_neighbor_non_zero) >> 2) |
                (atable(i, j + 7 + right_neighbor_non_zero) >> 2) |
                _mm256_movemask_epi8(_mm256_cmpgt_epi32(input, three)) |
                _mm256_movemask_epi8(_mm256_cmpgt_epi32(input_top, three)) |
                _mm256_movemask_epi8(_mm256_cmpgt_epi32(input_bottom, three));
        }
    }

    return diff;
}

/**
 * 4.5.2: Make use of explicit-vectorization to optimize the
 * asandPile_do_tile_default() function. This version
 * looks at all tiles and uses AVX512
 *
 * usage
 * ./run -k asandPile -s 256 -wt avx5 -m
 */
int asandPile_do_tile_avx5(int x, int y, int width, int height) {
    // all tiles are calculated with avx
    //  assumption DIM is divisible by 8
    int diff = 0;
    x = x * (x != 1);
    width =
        width + (x + TILE_W == DIM) +
        (x == 0); // if tile is not divisible by 8 we include the outer border
    height = height + (y + TILE_H == DIM) + (y == 0);

    __m512i input, input_top, input_bottom, three, DIV, shift_left, shift_right;
    three = _mm512_set1_epi32(3);

    for (int i = y; i < y + height && i < DIM - 1; i++) {
        for (int j = x; j < x + width; j += 16) {
            // we check if the left and right neighbor exists and if they should
            // be zero or not
            int left_neighbor_non_zero = j > 1;
            int right_neighbor_non_zero = j + 16 < (DIM - 1);
            // load data in vectors, if neighbor line doesn't exist, we load the
            // same line

            input = _mm512_loadu_epi32(&atable(i, j));
            input_top = _mm512_loadu_epi32(&atable(i - 1, j));
            input_bottom = _mm512_loadu_epi32(&atable(i + 1, j));

            // DIV 4
            DIV = _mm512_srli_epi32(input, 2);

            shift_left = _mm512_alignr_epi32(_mm512_setzero_epi32(), DIV, 1);
            shift_right = _mm512_alignr_epi32(DIV, _mm512_setzero_epi32(), 15);

            // add and modulo
            input = _mm512_add_epi32(
                _mm512_add_epi32(_mm512_and_epi32(input, three), shift_left),
                shift_right);

            input_top = _mm512_add_epi32(
                input_top, _mm512_maskz_add_epi32(0xffff * (i > 1), DIV,
                                                  _mm512_setzero_epi32()));
            input_bottom = _mm512_add_epi32(
                input_bottom,
                _mm512_maskz_add_epi32(0xffff * (i < DIM - 2), DIV,
                                       _mm512_setzero_epi32()));

            // store if neighbors are existant
            TYPE div[16];
            _mm512_storeu_epi32(&div, DIV);

            atable(i, j - left_neighbor_non_zero) += div[0];
            atable(i, j + 15 + right_neighbor_non_zero) += div[15];

            TYPE top[16];
            _mm512_storeu_epi32(&top, input_top);

            TYPE current[16];
            _mm512_storeu_epi32(&current, input);

            TYPE bottom[16];
            _mm512_storeu_epi32(&bottom, input_bottom);

            atable(i - 1, j + 1 - left_neighbor_non_zero) = top[0];
            atable(i, j + 1 - left_neighbor_non_zero) = current[0];
            atable(i + 1, j + 1 - left_neighbor_non_zero) = bottom[0];

            atable(i - 1, j + 14 + right_neighbor_non_zero) = top[15];
            atable(i, j + 14 + right_neighbor_non_zero) = current[15];
            atable(i + 1, j + 14 + right_neighbor_non_zero) = bottom[15];

            for (int k = 1; k < 15; k++) {
                atable(i - 1, j + k) = top[k];
                atable(i + 1, j + k) = bottom[k];
                atable(i, j + k) = current[k];
            }

            // check if one of them is still above 3
            diff |= (atable(i, j - left_neighbor_non_zero) >> 2) |
                    (atable(i, j + 15 + right_neighbor_non_zero) >> 2) |
                    _mm512_cmpgt_epi32_mask(input, three) |
                    _mm512_cmpgt_epi32_mask(input_top, three) |
                    _mm512_cmpgt_epi32_mask(input_bottom, three);
        }
    }

    return diff;
}

#endif /* __AVX512F__ */

#pragma GCC pop_options

unsigned asandPile_compute_seq(unsigned nb_iter) {
    int change = 0;
    for (unsigned it = 1; it <= nb_iter; it++) {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        change = do_tile(1, 1, DIM - 2, DIM - 2, 0);

        if (change == 0)
            return it;
    }
    return 0;
}

unsigned asandPile_compute_tiled(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W) {
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  0 /* CPU id */);
            }
        if (change == 0)
            return it;
    }
    return 0;
}

/*
 *
 * 4.3: An OMP implementation of the asynchronous version:
 * To avoid conflict with tiles the grid is computed as a a checkerboard
 * threads are doing the first color of the checkerboard then the second
 *
 * usage:
 * ./run -k asandPile -s 256 -v omp -wt opt -m
 *
 *        --- Color blue : ---
 *
 *
 *     --------------------------
 *     | b | _ | b | _ | b | _ |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | b | _ | b | _ | b | _ |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | b | _ | b | _ | b | _ |
 *     --------------------------
 *
 *        --- Color red : ---
 *
 *     --------------------------
 *     | _ | r | _ | r | _ | r |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | _ | r | _ | r | _ | r |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | _ | r | _ | r | _ | r |
 *     --------------------------
 *
 *
 *       --- Color green : ---
 *
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | g | _ | g | _ | g | _ |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | g | _ | g | _ | g | _ |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *
 *
 *       --- Color yellow : ---
 *
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | _ | y | _ | y | _ | y |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *     | _ | y | _ | y | _ | y |
 *     --------------------------
 *     | _ | _ | _ | _ | _ | _ |
 *     --------------------------
 *
 *
 */

unsigned asandPile_compute_omp_tiled(unsigned nb_iter) {

    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;
        // color blue
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H) {
            for (int x = 0; x < DIM; x += 2 * TILE_W) {
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  omp_get_thread_num());
            }
        }
// color red
#pragma omp parallel for schedule(runtime) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H) {
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W) {
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  omp_get_thread_num());
            }
        }

// color green
#pragma omp parallel for schedule(runtime) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H) {
            for (int x = 0; x < DIM; x += 2 * TILE_W) {
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  omp_get_thread_num());
            }
        }

// color yellow
#pragma omp parallel for schedule(runtime) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H) {
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W) {
                change |= do_tile(x + (x == 0), y + (y == 0),
                                  TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                  TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                  omp_get_thread_num());
            }
        }
        if (change == 0) {
            return it;
        }
    }
    return 0;
}

/**
 * 4.3: Horrible way to implement asynchronous with OMP,
 * All shared cell have a mutex
 * + an extra cost of if else branch...
 *
 * Really bad performance
 *
 *
 */
int asandPile_do_tile_mutex(int x, int y, int width, int height) {
    int change = 0;
    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            if (atable(i, j) >= 4) {
                if (j - 1 <= x) {
#pragma omp critical
                    atable(i, j - 1) += atable(i, j) / 4;
                } else {
                    atable(i, j - 1) += atable(i, j) / 4;
                }
                if (j + 1 >= x + width - 1) {
#pragma omp critical
                    atable(i, j + 1) += atable(i, j) / 4;
                } else {
                    atable(i, j + 1) += atable(i, j) / 4;
                }

                if (i - 1 <= y) {
#pragma omp critical
                    atable(i - 1, j) += atable(i, j) / 4;
                } else {
                    atable(i - 1, j) += atable(i, j) / 4;
                }
                if (i + 1 >= y + height - 1) {
#pragma omp critical
                    atable(i + 1, j) += atable(i, j) / 4;
                } else {
                    atable(i + 1, j) += atable(i, j) / 4;
                }
                if (i <= x || x >= x + width - 1 || j <= y ||
                    y >= y + height - 1) {
#pragma omp critical
                    atable(i, j) %= 4;
                } else {
                    atable(i, j) %= 4;
                }
                change = 1;
            }
        }

    return change;
}

/**
 * 4.3: Horrible way to implement asynchronous with OMP,
 * All shared cell have a mutex
 * + an extra cost of if else branch...
 *
 * Really bad performance
 *
 *
 */
unsigned asandPile_compute_omp_mutex(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel for
        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W) {
                int startx = x + (x == 0);
                int starty = y + (y == 0);
                int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
                int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));
                change |= asandPile_do_tile_mutex(startx, starty, tile_width,
                                                  tile_height);
            }
        if (change == 0)
            return it;
    }

    return 0;
}

/**
 * This function checks if a tile is unstable.
 * A tile is unstable if a neigbouring the tile that contains 4 or more
 * particles of sand.
 *
 * If the tile is unstable it is computed and the function updates the state
 * of the neigbouring tiles
 *
 * let's look at an example
 *
 *
 * EXAMPLE :
 *
 * --- Before Computation ---
 *
 * `Stable tile 3x3 (A)`               `Unstable tile 3x3 (B) `
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 0 | 2 | 0 |
 *     -------------   in contact with    -------------
 *     | 0 | 0 | 0 | ---------------------| 16| 3 | 1 |
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 0 | 1 | 2 |
 *     -------------                      -------------
 *
 * --- After Computation ---
 *
 * `Stable tile 3x3 (A)`               `Unstable tile 3x3 (B) `
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 4 | 5 | 0 |
 *     -------------   in contact with    -------------
 *     | 0 | 0 | 4 | ---------------------| 0 | 7 | 1 |
 *     -------------                      -------------
 *     | 0 | 0 | 0 |                      | 4 | 1 | 2 |
 *     -------------                      -------------
 *
 * In this example we see that after computation a stable Tile (A) can be
 * modified by computing a stable tile. So this function makes sure to
 * update the state of neighbouring tiles after computation.
 *
 * To do that from the border of the computed unstable tile (B), we check
 * all the neighbour cell that are outside of the Tile (B) so that they have
 * 4 or more grain of sand. If we find a neigbouring tile that got unstable
 * we mark it.
 *
 */
static int async_lazy_do_tile(int x, int y) {
    int uy = y / TILE_H;
    int ux = x / TILE_W;
    if (utable(ux, uy)) { // check if tile is unstable
                          //
        int startx = x + (x == 0);
        int starty = y + (y == 0);
        int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
        int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));

        utable(ux, uy) = do_tile(startx, starty, tile_width, tile_height,
                                 omp_get_thread_num());
        // down
        if (uy > 0) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (atable(starty - 1, i) >= 4)
                    utable(ux, uy - 1) = 1;
            }
        }
        // up
        if (uy < U_HEIGHT - 1) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (atable(starty + tile_height, i) >= 4)
                    utable(ux, uy + 1) = 1;
            }
        }
        // left
        if (ux > 0) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (atable(i, startx - 1) >= 4)
                    utable(ux - 1, uy) = 1;
            }
        }
        // down
        if (ux < U_WIDTH - 1) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (atable(i, startx + tile_width) >= 4)
                    utable(ux + 1, uy) = 1;
            }
        }
    }
    return utable(ux, uy);
}

/**
 * 4.4: Lazy  implementation asynchronous.
 * To be able to have cpu to work only where it needs
 * the tiles are marked as unstable or unstable with the utable(),
 *
 * [see UNSTABLE_TILES]
 *
 * The call do_tile is therefore only done on unstable tiles
 *
 * usage:
 * ./run -k asandPile -s 512 -wt opt -v -a alea lazy -m
 *
 */
unsigned asandPile_compute_lazy(unsigned nb_iter) {
    int change = 0;
    for (unsigned it = 1; it <= nb_iter; it++) {

        // for each tiles
        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                change |= async_lazy_do_tile(x, y);
            }
        }
        if (change == 0)
            return it;
    }

    return 0;
}

/**
 * 4.4: Lazy OMP implementation asynchronous.
 * To be able to have cpu to work only where it needs
 * the tiles are marked as unstable or unstable with the utable(),
 *
 * [see UNSTABLE_TILES]
 * The call do_tile is therefore only done on unstable tiles
 *
 * To resolve conflict beetween tiles the strategy has been use than for
 * ssandPile_compute_omp()
 * The use of a checkerboard to distribute threads
 *
 *
 * usage:
 * ./run -k asandPile -s 512 -wt opt -v -a alea omp_lazy -m
 *
 */
unsigned asandPile_compute_omp_lazy(unsigned nb_iter) {
    int change = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {

        // blue
#pragma omp parallel for schedule(runtime) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H) {
            for (int x = 0; x < DIM; x += 2 * TILE_W) {
                change |= async_lazy_do_tile(x, y);
            }
        }

        // color red
#pragma omp parallel for schedule(runtime) reduction(| : change)

        for (int y = 0; y < DIM; y += 2 * TILE_H) {
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W) {
                change |= async_lazy_do_tile(x, y);
            }
        }

        // color green
#pragma omp parallel for schedule(runtime) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H) {
            for (int x = 0; x < DIM; x += 2 * TILE_W) {
                change |= async_lazy_do_tile(x, y);
            }
        }

        // color yellow
#pragma omp parallel for schedule(runtime) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H) {
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W) {
                change |= async_lazy_do_tile(x, y);
            }
        }
        if (change == 0)
            return it;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////// Not necessarly good or useful ideas with purpose of
/// optimization
///////////
/////////////////////////////////////////////////////////////////////////////////////////

static inline int async_default_lazy_do_tile(int x, int y, int width,
                                             int height) {
    int ux = x / TILE_W;
    int uy = y / TILE_H;
    if (utable(ux, uy)) { // check if tile is unstable
        int startx = x + (x == 0);
        int starty = y + (y == 0);
        int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
        int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));

        utable(ux, uy) = do_tile(startx, starty, tile_width, tile_height, 0);

        if (uy > 0) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (atable(starty - 1, i) >= 4)
                    utable(ux, uy - 1) = 1;
            }
        }
        if (uy < height - 1) {
            for (int i = startx; i < startx + tile_width; i++) {
                if (atable(starty + tile_height, i) >= 4)
                    utable(ux, uy + 1) = 1;
            }
        }
        if (ux > 0) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (atable(i, startx - 1) >= 4)
                    utable(ux - 1, uy) = 1;
            }
        }
        if (ux < width - 1) {
            for (int i = starty; i < starty + tile_height; i++) {
                if (atable(i, startx + tile_width) >= 4)
                    utable(ux + 1, uy) = 1;
            }
        }
    }
    return utable(ux, uy);
}

static inline int async_opt_lazy_do_tile(int x, int y, int width, int height) {
    int ux = x / TILE_W;
    int uy = y / TILE_H;
    if (utable(ux, uy)) { // check if tile is unstable
        int startx = x + (x == 0);
        int starty = y + (y == 0);
        int tile_width = TILE_W - ((x + TILE_W == DIM) + (x == 0));
        int tile_height = TILE_H - ((y + TILE_H == DIM) + (y == 0));

        utable(ux, uy) = do_tile(startx, starty, tile_width, tile_height, 0);
        if (uy > 0 || uy < height - 1 || ux > 0 || ux < width - 1)
            return async_default_lazy_do_tile(x, y, width, height);

        for (int i = startx; i < startx + tile_width; i++) {
            if (atable(starty - 1, i) >= 4)
                utable(ux, uy - 1) = 1;
            if (atable(starty + tile_height, i) >= 4)
                utable(ux, uy + 1) = 1;
        }
        for (int i = starty; i < starty + tile_height; i++) {
            if (atable(i, startx - 1) >= 4)
                utable(ux - 1, uy) = 1;
            if (atable(i, startx + tile_width) >= 4)
                utable(ux + 1, uy) = 1;
        }
    }
    return utable(ux, uy);
}

/**
 * Kind of horrible implementation with purpose to improve performance
 * of the asynchronous lazy sequential function...
 *
 * To try gainning speed here the function async_opt_lazy_do_tile() do a
 * call to async_default_lazy_do_tile() only if it's a tile on the border of
 * the grid
 *
 * Else it is executing 2 for loop instead of for and avoid to have
 * 4 if checks for every call
 *
 * usage:
 * ./run -k asandPile -v lazy2 -wt opt -a alea -s 512 -m
 *
 *
 */
unsigned asandPile_compute_lazy2(unsigned nb_iter) {
    // fprintf(stderr, "Executing asandPile_compute_lazy Tile \n");
    int change = 0;

    int width = DIM / TILE_W;
    int height = DIM / TILE_H;
    for (unsigned it = 1; it <= nb_iter; it++) {

        // for each tiles
        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                change |= async_opt_lazy_do_tile(x, y, width, height);
            }
        }
        if (change == 0)
            return it;
    }

    return 0;
}
