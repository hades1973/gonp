// Package blas provides a interface for blas
// go install .
package blas

/*
#cgo CFLAGS: -I/home/gns/thirdparty/include
#cgo LDFLAGS: -L/home/gns/thirdparty/lib -lopenblas

#include <cblas.h>
*/
import "C"

const (
	RowMajor    int32 = 101
	ColMajor    int32 = 102
	NoTrans     int32 = 111
	Trans       int32 = 112
	ConjTrans   int32 = 113
	ConjNoTrans int32 = 114
	Upper       int32 = 121
	Lower       int32 = 122
	NonUnit     int32 = 131
	Unit        int32 = 132
	Left        int32 = 141
	Right       int32 = 142
)
