// Package blas provides a interface for blas
// go install .
package blas

import (
	"unsafe"
)

/*
#cgo CFLAGS: -I/home/gns/thirdparty/include
#cgo LDFLAGS: -L/home/gns/thirdparty/lib -lopenblas

#include <cblas.h>
*/
import "C"

////////////////////////////////////////////
//
// blas level 1 api
//
////////////////////////////////////////////

// Ddot: dot <- x^T \times y
func Ddot(x, y Vector) float64 {
	var dot C.double
	dot = C.cblas_ddot(
		(C.blasint)(x.N),
		(*C.double)(unsafe.Pointer(&(x.Base[x.Off]))), (C.blasint)(x.Inc),
		(*C.double)(unsafe.Pointer(&(y.Base[y.Off]))), (C.blasint)(y.Inc),
	)

	return float64(dot)
}

// Daxpy: y <- alpha \times x + y
func Daxpy(alpha float64, x, y Vector) {
	C.cblas_daxpy(
		(C.blasint)(x.N),
		(C.double)(alpha),
		(*C.double)(unsafe.Pointer(&(x.Base[x.Off]))), (C.blasint)(x.Inc),
		(*C.double)(unsafe.Pointer(&(y.Base[y.Off]))), (C.blasint)(y.Inc),
	)
}

// Dscal: x <- alpha \times x
func Dscal(alpha float64, x Vector) {
	C.cblas_dscal(
		(C.blasint)(x.N),
		(C.double)(alpha),
		(*C.double)(unsafe.Pointer(&(x.Base[x.Off]))), (C.blasint)(x.Inc),
	)
}

// Dswap: x <-> y
func Dswap(x, y Vector) {
	C.cblas_dswap(
		C.blasint(x.N),
		(*C.double)(unsafe.Pointer(&(x.Base[x.Off]))), (C.blasint)(x.Inc),
		(*C.double)(unsafe.Pointer(&(y.Base[y.Off]))), (C.blasint)(y.Inc),
	)
}
