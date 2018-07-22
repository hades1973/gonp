package main

import (
	"fmt"
	"gonp/blas"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func prtM(A *blas.GeMatrix) {
	for i := 0; i < A.M; i++ {
		for j := 0; j < A.N; j++ {
			fmt.Printf("%6.1f", A.At(i, j))
		}
		fmt.Println()
	}
}

func main() {
	m, n := 10000, 10000
	A := mat.NewDense(m, n, make([]float64, m*n, m*n))
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			A.Set(i, j, rand.Float64())
		}
	}

	v := mat.NewVecDense(m, nil)
	for i := 0; i < m; i++ {
		v.SetVec(i, rand.Float64())
	}

	u := mat.NewVecDense(m, nil)

	tm := time.Now()
	for i := 0; i < 10; i++ {
		u.MulVec(A, v)
	}
	fmt.Println(time.Since(tm).Seconds() / 10.0)
}
