// Package lapacke provides a interface for lapack
// go install .
package lapack

/*
#cgo CFLAGS: -I/home/gns/include
#cgo LDFLAGS: -L/home/gns/lib -lopenblas

#include <lapacke.h>
lapack_int LAPACKE_dgetrs (int matrix_layout , char trans , lapack_int n , lapack_int nrhs , const double * a , lapack_int lda , const lapack_int * ipiv , double * b , lapack_int ldb );

lapack_int LAPACKE_dgetri (int matrix_layout , lapack_int n , double * a , lapack_int lda , const lapack_int * ipiv );
*/
import "C"

import (
	"gonp/blas"
	"unsafe"
)

// Dgetri solve A^(-1), after factorize matrix A with Dgetrf.
func Dgetri(A *blas.GeMatrix, ipiv []int32) int {
	var info C.lapack_int
	info = C.LAPACKE_dgetri(
		C.lapack_int(blas.RowMajor), // 行序存储
		(C.lapack_int)(A.M),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])), (C.lapack_int)(A.LD),
		(*C.lapack_int)(unsafe.Pointer(&ipiv[0])),
	)
	return int(info)
}

// Dgetrs solve AX=B, after factorize matrix A with Dgetrf.
func Dgetrs(A, B *blas.GeMatrix, ipiv []int32) int {
	var info C.lapack_int
	info = C.LAPACKE_dgetrs(
		C.lapack_int(blas.RowMajor), // 行序存储
		(C.char)('n'),
		(C.lapack_int)(A.M), (C.lapack_int)(B.N),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])), (C.lapack_int)(A.LD),
		(*C.lapack_int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&B.Base[B.O_i*B.LD+B.O_j])), (C.lapack_int)(B.LD),
	)
	return int(info)
}

// Dgetrf factorize matrix A.
// A = P * L * U
// P is a permutation matrix, use ipiv vector in subroutine.
// L is lower triangular with unit diagonal elements (lower trapezoidal if m > n)
// U is upper triangular (upper trapezoidal if m <n).
// L, U stores in A
// The routine uses partial pivoting, with row interchanges.
// ipiv size is max(1, min(A.M, A.N))
func Dgetrf(A *blas.GeMatrix, ipiv []int32) int {
	var info C.lapack_int
	info = C.LAPACKE_dgetrf(
		C.lapack_int(blas.RowMajor), // 行序存储
		(C.lapack_int)(A.M), (C.lapack_int)(A.N),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])), (C.lapack_int)(A.LD),
		(*C.lapack_int)(unsafe.Pointer(&ipiv[0])),
	)
	return int(info)
}

// Dgels solves The Linear Least Squares Problem
// minimize X for ||AX-B||_2
// the X put into B
// storage is row major, but matrix look as  column vectors.
func Dgels(trans int, A, B *blas.GeMatrix) int {
	var info C.lapack_int
	info = C.LAPACKE_dgels(C.lapack_int(blas.RowMajor), // 行序存储
		(C.char)(trans),
		(C.lapack_int)(A.M), (C.lapack_int)(A.N), C.int(B.N),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])), (C.lapack_int)(A.LD),
		(*C.double)(unsafe.Pointer(&B.Base[B.O_i*B.LD+B.O_j])), (C.lapack_int)(B.LD))
	return int(info)
}

/*
Description
	The routine solves for X the system of linear equations A*X = B,
	where A is an n-by-n matrix,
	the columns of matrix B are individual right-hand sides,
	and the columns of X are the corresponding solutions.

	The LU decomposition with partial pivoting and row interchanges is used to factor A as A = P*L*U,
	where P is a permutation matrix,
		  L is unit lower triangular,
	  and U is upper triangular.
	The factored form of A is then used to solve the system of equations A*X = B.
Input Parameters
	matrix_layout
		Specifies whether matrix storage layout is row major (LAPACK_ROW_MAJOR) or column major (LAPACK_COL_MAJOR).
	n
		The number of linear equations, that is, the order of the matrix A; n≥ 0.
	nrhs
		The number of right-hand sides, that is, the number of columns of the matrix B; nrhs≥ 0.
	a
		The array a(size max(1, lda*n)) contains the n-by-n coefficient matrix A.
	b
		The array b of size max(1, ldb*nrhs) for column major layout
		and max(1, ldb*n) for row major layout contains the n-by-nrhs matrix of right hand side matrix B.
	lda
		The leading dimension of the array a; lda≥ max(1, n).
	ldb
		The leading dimension of the array b; ldb≥ max(1, n) for column major layout
		and ldb≥nrhs for row major layout.
	ldx
		The leading dimension of the array x; ldx≥ max(1, n) for column major layout
		and ldx≥nrhs for row major layout.
Output Parameters
	a
		Overwritten by the factors L and U from the factorization of A = P*L*U;
		the unit diagonal elements of L are not stored.
		If iterative refinement has been successfully used (info= 0 and iter≥ 0), then A is unchanged.
		If double precision factorization has been used (info= 0 and iter < 0),
		then the array A contains the factors L and U from the factorization A = P*L*U;
		the unit diagonal elements of L are not stored.
	b
		Overwritten by the solution matrix X for dgesv, sgesv,zgesv,zgesv. Unchanged for dsgesv and zcgesv.
	ipiv
		Array, size at least max(1, n).
		The pivot indices that define the permutation matrix P;
		row i of the matrix was interchanged with row ipiv[i-1].
		Corresponds to the single precision factorization (if info= 0 and iter≥ 0)
		or the double precision factorization (if info= 0 and iter < 0).
	x
		Array, size max(1, ldx*nrhs) for column major layout and max(1, ldx*n) for row major layout.
		If info = 0, contains the n-by-nrhs solution matrix X.
	iter
		If iter < 0: iterative refinement has failed, double precision factorization has been performed
		If iter = -1: the routine fell back to full precision for implementation- or machine-specific reason
		If iter = -2: narrowing the precision induced an overflow, the routine fell back to full precision
		If iter = -3: failure of sgetrf for dsgesv, or cgetrf for zcgesv
		If iter = -31: stop the iterative refinement after the 30th iteration.
		If iter > 0: iterative refinement has been successfully used. Returns the number of iterations.
Return Values
	This function returns a value info.
	If info=0, the execution is successful.
	If info = -i, parameter i had an illegal value.
	If info = i, Ui, i (computed in double precision for mixed precision subroutines) is exactly zero.
	The factorization has been completed, but the factor U is exactly singular, so the solution could not be computed.
*/
func Dgesv(trans int, A *blas.GeMatrix, ipiv []int, B *blas.GeMatrix) int {
	var info C.int
	var n int
	if A.M > A.N {
		n = A.N
	} else {
		n = A.M
	}
	info = C.LAPACKE_dgesv(C.lapack_int(blas.RowMajor),
		C.lapack_int(n),
		C.lapack_int(B.N),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])), C.lapack_int(A.LD),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&B.Base[B.O_i*B.LD+B.O_j])), C.lapack_int(B.LD),
	)
	return int(info)
}

/*
Description
	The routine computes all eigenvalues and,
	optionally, eigenvectors of a real symmetric matrix A.
	Note that for most cases of real symmetric eigenvalue problems
	the default choice should be syevr function as its underlying algorithm is faster and uses less workspace.
Input Parameters
	matrix_layout
		Specifies whether matrix storage layout is row major (LAPACK_ROW_MAJOR) or column major (LAPACK_COL_MAJOR).
	jobz
		Must be 'N' or 'V'.
		If jobz = 'N', then only eigenvalues are computed.
		If jobz = 'V', then eigenvalues and eigenvectors are computed.
	uplo
		Must be 'U' or 'L'.
		If uplo = 'U', a stores the upper triangular part of A.
		If uplo = 'L', a stores the lower triangular part of A.
	n
		The order of the matrix A (n≥ 0).
	a
		a (size max(1, lda*n)) is an array containing either upper or lower triangular part of the symmetric matrix A,
		as specified by uplo.
	lda
		The leading dimension of the array a. Must be at least max(1, n).
Output Parameters
	a
		On exit, if jobz = 'V', then if info = 0, array a contains the orthonormal eigenvectors of the matrix A.
		If jobz = 'N', then on exit the lower triangle (if uplo = 'L') or the upper triangle (if uplo = 'U') of A,
		including the diagonal, is overwritten.
	w
		Array, size at least max(1, n). If info = 0, contains the eigenvalues of the matrix A in ascending order.
Return Values
	This function returns a value info.
	If info=0, the execution is successful.
	If info = -i, the i-th parameter had an illegal value.
	If info = i, then the algorithm failed to converge;
	i indicates the number of elements of an intermediate tridiagonal form which did not converge to zero.
*/
// func Dsyev(jobz byte, uplo byte, n int, a []float64, lda int, w []float64) int {
// 	var info C.int
// 	info = C.LAPACKE_dsyev(RowMajor,
// 		C.char(jobz),
// 		C.char(uplo),
// 		C.lapack_int(n),
// 		(*C.double)(unsafe.Pointer(&a[0])), C.lapack_int(lda),
// 		(*C.double)(unsafe.Pointer(&w[0])))
// 	return int(info)
// }
