// Package blas provides a interface for blas
// go install .
package blas

/*
#cgo CFLAGS: -I/home/gns/thirdparty/include
#cgo LDFLAGS: -L/home/gns/thirdparty/lib -lopenblas

#include <cblas.h>
*/
import "C"
import "unsafe"

////////////////////////////////////////////
//
// blas level 3 api
//
////////////////////////////////////////////

func Dgemm(transA, transB int32, alpha float64, A *GeMatrix, B *GeMatrix, beta float64, CC *GeMatrix) {
	C.cblas_dgemm(
		C.enum_CBLAS_ORDER(RowMajor),
		C.enum_CBLAS_TRANSPOSE(transA),
		C.enum_CBLAS_TRANSPOSE(transB),
		C.blasint(A.M), C.blasint(A.N), C.blasint(B.N),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])),
		C.blasint(A.LD),
		(*C.double)(unsafe.Pointer(&B.Base[B.O_i*B.LD+B.O_j])),
		C.blasint(B.LD),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&CC.Base[CC.O_i*CC.LD+CC.O_j])),
		C.blasint(CC.LD),
	)
}
