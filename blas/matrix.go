// Package blas provides a interface for blas
// go install .
package blas

/*
#cgo CFLAGS: -I/home/gns/thirdparty/include
#cgo LDFLAGS: -L/home/gns/thirdparty/lib -lopenblas

#include <cblas.h>
*/
import "C"

// Gematrix represents a general matrix using the conventional storage scheme.
type GeMatrix struct {
	M, N     int // M x N elements, M is rows, N is cols
	LD       int // from A[i,j] to A[i+1, j] nees step LD elements
	O_i, O_j int // first element is Base[O_i * LD + O_j]
	Base     []float64
}

// At return A[i, j]
func (A *GeMatrix) At(i, j int) float64 {
	return A.Base[(i+A.O_i)*A.LD+(j+A.O_j)]
}

// Set set A[i, j] to val, and return val
func (A *GeMatrix) Set(i, j int, val float64) float64 {
	A.Base[(i+A.O_i)*A.LD+(j+A.O_j)] = val
	return val
}

// Row return ith row vector of A.
// Any modify element in vector will affect A,
// for vector and A own same memory block
func (A *GeMatrix) Row(i int) *Vector {
	return &Vector{
		N:    A.N,
		Inc:  1,
		Off:  (i+A.O_i)*A.LD + A.O_j,
		Base: A.Base,
	}
}

// Col return ith col vector of A.
// Any modify element in vector will affect A,
// for vector and A own same memory block
func (A *GeMatrix) Col(j int) *Vector {
	return &Vector{
		N:    A.M,
		Inc:  A.LD,
		Off:  A.O_i*A.LD + A.O_j + j,
		Base: A.Base,
	}
}

// Band represents a band matrix using the band storage scheme.
type Band struct {
	Rows, Cols int
	KL, KU     int
	LD         int
	Data       []float64
}

// Triangular represents a triangular matrix using the conventional storage scheme.
type Triangular struct {
	N    int
	LD   int
	Data []float64
	//	Uplo blas.Uplo
	//	Diag blas.Diag
}

// TriangularBand represents a triangular matrix using the band storage scheme.
type TriangularBand struct {
	N, K int
	LD   int
	Data []float64
	//	Uplo blas.Uplo
	//	Diag blas.Diag
}

// TriangularPacked represents a triangular matrix using the packed storage scheme.
type TriangularPacked struct {
	N    int
	Data []float64
	//	Uplo blas.Uplo
	//	Diag blas.Diag
}

// Symmetric represents a symmetric matrix using the conventional storage scheme.
type Symmetric struct {
	N    int
	LD   int
	Data []float64
	//	Uplo blas.Uplo
}

// SymmetricBand represents a symmetric matrix using the band storage scheme.
type SymmetricBand struct {
	N, K int
	LD   int
	Data []float64
	//	Uplo blas.Uplo
}

// SymmetricPacked represents a symmetric matrix using the packed storage scheme.
type SymmetricPacked struct {
	N    int
	Data []float64
	//	Uplo blas.Uplo
}
