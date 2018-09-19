#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

int main(){

  int shapea = 320;
  int shapeb = 100;
  int shapec = 8;
  int shaped = 64; 
  int memlen = shapea* shapeb* shapec * shaped;
  
  float* indptr_ = (float*) malloc( memlen*sizeof( float ) );
  float* outdptr_ = (float*) malloc( memlen*sizeof( float ) );
   
  int l ;
  float u = 0;
  for ( l = 0; l < memlen; l++) {
        indptr_[l] = u;
        u++;
     
  }
  

  int src_shape[] = {shapea, shapeb, shapec, shaped};
  int dst_shape[] = {shapea, shapec, shapeb, shaped};
  int src_stride[4];
  int dst_stride[4];
 
  src_stride[3] = 1;
  dst_stride[3] = 1;
  int p;
  for (p = 2; p >= 0; --p) {
    src_stride[p] = src_shape[p + 1] * src_stride[p + 1];
    dst_stride[p] = dst_shape[p + 1] * dst_stride[p + 1];
  }

  int n = src_shape[1];
  int m = src_shape[2];
  int block_size ;

  for (block_size=5;block_size<10;block_size++){
    printf("block_size,%d,%f,\n", block_size,indptr_[1]);
    long start, end;
    struct timeval timecheck;

    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

      #pragma omp parallel for
        int k;
        for (k = 0; k < dst_shape[0]; k++) {
          int indexstride = k*dst_stride[0];

    //transpose 3D matrix by block
    int i;
    for(i=0; i<n; i+=block_size) {
      int max_i = (i+block_size) < n ? block_size : (n-i);
      int A = indexstride + i * src_stride[1];
      int B = indexstride + i * dst_stride[2];
      int j;
      for (j = 0; j < m; j += block_size) {
        int max_j = (j + block_size) < m ? block_size : (m - j);
        // block operation
        int bi;
        for ( bi = 0; bi < max_i; bi++) {
          int AA = A + bi * src_stride[1];
          int BB = B + bi * dst_stride[2];
          int bj;
          for ( bj = 0; bj < max_j; bj++) {
            int c;
            for (c = 0; c < src_shape[3]; c++) {
              outdptr_[BB + c] = indptr_[AA + c];
            }

            AA += src_stride[2];
            BB += dst_stride[1];
          }
        }
        A+= block_size*src_stride[2];
        B += block_size * dst_stride[1];
      }
    }

  }
  gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    printf("%ld milliseconds elapsed\n", (end - start));
     
  }

 
/*
  printf(" %f : %f -- %f : %f \n" ,indptr_[1 * src_stride[0] + (src_shape[1]-1)*src_stride[1]+(src_shape[2]-1)*src_stride[2]],
    outdptr_[1 * dst_stride[0] + (dst_shape[1]-1)*dst_stride[1]+(dst_shape[2]-1)*dst_stride[2]],
    indptr_[1 * src_stride[0] + (src_shape[1]-1)*src_stride[1]+(src_shape[2]-1)*src_stride[2]+2],
    outdptr_[1 * dst_stride[0] + (dst_shape[1]-1)*dst_stride[1]+(dst_shape[2]-1)*dst_stride[2]+2]);
*/
/*
  std::cout << indptr_[1 * src_stride[0] + (src_shape[1]-1)*src_stride[1]+(src_shape[2]-1)*src_stride[2]] <<":"
            << outdptr_[1 * dst_stride[0] + (dst_shape[1]-1)*dst_stride[1]+(dst_shape[2]-1)*dst_stride[2]] <<"--"
            << indptr_[1 * src_stride[0] + (src_shape[1]-1)*src_stride[1]+(src_shape[2]-1)*src_stride[2]+2] <<":"
            << outdptr_[1 * dst_stride[0] + (dst_shape[1]-1)*dst_stride[1]+(dst_shape[2]-1)*dst_stride[2]+2]
            << std::endl;
*/
  free(indptr_);
  free(outdptr_);
  return 0;
}

