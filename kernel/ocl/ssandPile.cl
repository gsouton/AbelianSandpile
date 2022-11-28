#include "kernel/ocl/common.cl"


__kernel void ssandPile_ocl (__global unsigned *in, __global unsigned *out) {
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

}


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
__kernel void ssandPile_ocl_hybrid (__global unsigned *in,
                                    __global unsigned *out,
                                    __global unsigned *buffer,
                                    unsigned gpu_y_end,
                                    unsigned valid_copied_lines) {
  int x = get_global_id (0);
  int y = get_global_id (1);
  // if(x == 0 && y == 0)
  //   printf("\tgpu_end: %d\n", gpu_y_end);
  if(y < gpu_y_end + valid_copied_lines){

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
}


// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a ssandPile-specific version (generic version is defined in common.cl)
__kernel void ssandPile_update_texture (__global unsigned *cur, __write_only image2d_t tex) {
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c = cur [y * DIM + x];
  unsigned r = 0, v = 0, b = 0;

  if (c == 1)
    v = 255;
  else if (c == 2)
    b = 255;
  else if (c == 3)
    r = 255;
  else if (c == 4)
    r = v = b = 255;
  else if (c > 4)
    r = v = b = (2 * c);

  c = rgba(r, v, b, 0xFF);
  write_imagef (tex, pos, color_scatter (c));
}
