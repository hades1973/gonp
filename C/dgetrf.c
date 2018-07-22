/* Calling DGELS using row-major order */
// gcc main.c -I/home/gns/include -L/home/gns/lib -lopenblas

#include <stdio.h>
#include <lapacke.h>

void prtM(int m, int n, double *A);
void prtV(int n, int *v);

int main (int argc, const char * argv[])
{
   double A[3][3] = {
	   10.0, -7.0, 0.0,
	   -3.0, 2.0, 6.0,
	   5.0, -1.0, 5.0,
	   };
   int ipiv[3] = {-1, -1, -1};
   int info, m, n, lda, ldb, nrhs;
   int i, j;

   m = 3;
   n = 3;
   lda = 3;

   prtM(3,3, &A[0][0]);
   info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, &A[0][0], lda, ipiv);
   prtM(3,3, &A[0][0]);

   prtV(3, ipiv);
}

void prtM(int m, int n, double *A) {
	int i, j;
   for(i=0;i<m;i++)
   {
      for(j=0;j<n;j++)
      {
         printf("%7.2f ",A[m*i + j]);
      }
      printf("\n");
   }
   printf("\n");
   printf("\n");
}

void prtV(int n, int *v) {
	for(int i = 0; i < n; i++) {
		printf("%3d\n", v[i]);
	}
	printf("\n");
	printf("\n");
}
