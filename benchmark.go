package main

import (
	"fmt"
	"gonp/blas"
	"math/rand"
	"time"
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
	var A = blas.GeMatrix{
		M: m, N: n, LD: n,
		OffRow: 0, OffCol: 0,
		Base: make([]float64, m*n),
	}
	var B = blas.GeMatrix{
		M: m, N: n, LD: n,
		OffRow: 0, OffCol: 0,
		Base: make([]float64, m*n),
	}
	var C = blas.GeMatrix{
		M: m, N: n, LD: n,
		OffRow: 0, OffCol: 0,
		Base: make([]float64, m*n),
	}

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			A.Set(i, j, rand.Float64())
			B.Set(i, j, rand.Float64())
		}
	}

	tm := time.Now()
	blas.Dgemm(blas.NoTrans, blas.NoTrans, 3.0, &A, &B, 1.0, &C)
	fmt.Println(time.Since(tm).Seconds() / 10.0)
}
