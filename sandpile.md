# Abelian Sandpile

This code can be found in kernel/c/sandPile.c

## Sequential computation

```C
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
```

## Sequential computation with compiler optimization

Deleting data dependency and branching

```C
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
```


## Using OpenMP

```C
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
```


## Using AVX

### AVX 256

```C
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
```


### AVX 512
```C

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
```


## Pure OpenCL implementation

Code on CPU side to invoke OpenCL kernel

```C
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
    }
    clFinish (queue);
    monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
    return 0;
}
```

OpenCL kernel
```C
__kernel void ssandPile_ocl_term (__global unsigned *in, __global unsigned *out, __global unsigned *buffer) {
  int x = get_global_id (0);
  int y = get_global_id (1);

  int tp_exist = y > 0;
  int bt_exist = y < (DIM-1);

  int l_exist = x > 0;
  int r_exist = x < (DIM-1);

  int current = y*DIM+x;
  int c_exist = tp_exist & bt_exist & l_exist & r_exist;  

  int top = ((y-tp_exist)*DIM+x);
  int bottom = ((y+bt_exist)*DIM+x);
  int left = (y*DIM+(x-l_exist));
  int right = (y*DIM+(x+r_exist));

  out[current] = (in[current]%4) * c_exist +
                 (in[top] / 4) * tp_exist + 
                 (in[bottom] / 4) * bt_exist + 
                 (in[left] / 4) * l_exist + 
                 (in[right] / 4) * r_exist;

  buffer[current] = out[current] / 4;
}
```

## Hybrid GPU (OpenCL) + CPU (AVX) implementation

```C
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
```

