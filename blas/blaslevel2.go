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
// blas level 2 api
//
////////////////////////////////////////////

func Dgemv(trans int, alpha float64, A *GeMatrix, x *Vector, beta float64, y *Vector) {
	C.cblas_dgemv(
		C.enum_CBLAS_ORDER(RowMajor),
		C.enum_CBLAS_TRANSPOSE(trans),
		C.blasint(A.M), C.blasint(A.N),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&A.Base[A.O_i*A.LD+A.O_j])),
		C.blasint(A.LD),
		(*C.double)(unsafe.Pointer(&x.Base[x.Off])),
		C.blasint(x.Inc),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&y.Base[y.Off])),
		C.blasint(y.Inc),
	)
}
