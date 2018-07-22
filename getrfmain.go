package main

import (
	"fmt"
	"gonp/blas"
	"gonp/lapack"
)

func main() {

	// |         A            |
	// -----------------------|
	// | 1, -2, 0, 0,  0,  0,|
	// |-1, +2, 1, 0,  0,  0,|
	// | 5, -1, 5, 0,  0,  0,|
	// | 0,  0, 0, 3, -1,  0,|
	// | 0,  0, 0, 2,  0, -1,|
	// | 0,  0, 0, 1, -1,  1,|
	// -----------------------
	var A = blas.GeMatrix{
		M: 3, N: 3, LD: 3,
		O_i: 0, O_j: 0,
		Base: []float64{
			10, -7, 0,
			-3, 2, 6,
			5, -1, 5,
		},
	}
	var ipiv = []int32{
		1, 2, 3,
	}

	fmt.Println("Matrix A")
	prtM(&A)
	fmt.Println()

	info := lapack.Dgetrf(&A, ipiv)
	fmt.Println("info: ", info)

	fmt.Println("Matrix A")
	prtM(&A)
	fmt.Println()
	fmt.Println("ipiv = ")
	for i := 0; i < len(ipiv); i++ {
		fmt.Printf("%d\n", ipiv[i])
	}
	fmt.Println()
	fmt.Println()

	var B = blas.GeMatrix{
		M: 3, N: 1, LD: 1,
		O_i: 0, O_j: 0,
		Base: []float64{
			7,
			4,
			6,
		},
	}
	lapack.Dgetrs(&A, &B, ipiv)
	prtM(&A)
	prtM(&B)
}

func prtM(A *blas.GeMatrix) {
	for i := 0; i < A.M; i++ {
		for j := 0; j < A.N; j++ {
			fmt.Printf("%7.2f", A.At(i, j))
		}
		fmt.Println()
	}
}
