package main

import (
	"fmt"
	"gonp/blas"
	"gonp/lapack"
)

func main() {
	// |     A     |    B      |
	// -----------------------
	// | 1, -2, 0, | 1, -2,  0,
	// |-1, +2, 1, | -1, 2,  1,
	// | 5, -1, 5, | 5, -1,  5
	// -----------------------
	var AA = blas.GeMatrix{
		M: 3, N: 6, LD: 6,
		O_i: 0, O_j: 0,
		Base: []float64{
			1, -2, 0, 1, -2, 0,
			-1, +2, 1, -1, 2, 1,
			5, -1, 5, 5, -1, 5,
		},
	}
	fmt.Println("Matrix A | A")
	prtM(&AA)
	fmt.Println()

	A := AA
	A.M, A.N = 3, 3

	B := AA
	B.M, B.N = 3, 3
	B.O_i, B.O_j = 0, 3

	var ipiv = []int32{1, 2, 3}
	var info int
	if info = lapack.Dgetrf(&B, ipiv); info != 0 {
		fmt.Println("Can't getri, error code: ", info)
	}
	if info = lapack.Dgetri(&B, ipiv); info != 0 {
		fmt.Println("Can't getri, error code: ", info)
	}
	fmt.Println("A|A^{-1} = ")
	prtM(&AA)
	fmt.Println()

	C := blas.GeMatrix{
		M: 3, N: 3, LD: 3,
		O_i: 0, O_j: 0,
		Base: []float64{
			0, 0, 0,
			0, 0, 0,
			0, 0, 0,
		},
	}
	// 验证A * A^{-1} = I
	blas.Dgemm(blas.NoTrans, blas.NoTrans, 1.0, &A, &B, 0.0, &C)
	fmt.Println("A * A{-1} = ")
	prtM(&C)
	fmt.Println()
}

func prtM(A *blas.GeMatrix) {
	for i := 0; i < A.M; i++ {
		for j := 0; j < A.N; j++ {
			fmt.Printf("%v  ", A.At(i, j))
		}
		fmt.Println()
	}
}
