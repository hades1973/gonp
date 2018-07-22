// Package blas provides a interface for blas
// go install .
package blas

/*
#cgo CFLAGS: -I/home/gns/thirdparty/include
#cgo LDFLAGS: -L/home/gns/thirdparty/lib -lopenblas

#include <cblas.h>
*/
import "C"

// Vector represents a vector with an associated element increment.
type Vector struct {
	N    int // N elements
	Inc  int // from v[i] to v[i+1] need step Inc elements
	Off  int // first element is Base[Off:]
	Base []float64
}

// At return v[i]
func (v *Vector) At(i int) float64 {
	return v.Base[v.Off+i*v.Inc]
}

// Set assign val to v[i], and return v[i]
func (v *Vector) Set(i int, val float64) float64 {
	v.Base[v.Off+i*v.Inc] = val
	return val
}
