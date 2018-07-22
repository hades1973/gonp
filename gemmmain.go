package main

import (
	"fmt"
	"gonp/blas"
)

func main() {
	// |     A     |    B      |
	// -----------------------
	// | 1, -2, 0, | 0,  0,  0,
	// |-1, +2, 1, | 0,  0,  0,
	// | 5, -1, 5, | 0,  0,  0,
	// | 0,  0, 0, | 3, -1,  0,
	// | 0,  0, 0, | 2,  0, -1,
	// | 0,  0, 0, | 1, -1,  1
	// -----------------------
	var AB = blas.GeMatrix{
		M: 6, N: 6, LD: 6,
		O_i: 0, O_j: 0,
		Base: []float64{
			1, -2, 0, 0, 0, 0,
			1, +2, 1, 0, 0, 0,
			5, -1, 5, 0, 0, 0,
			0, 0, 0, 3, -1, 0,
			0, 0, 0, 2, 0, -1,
			0, 0, 0, 1, -1, 1,
		},
	}
	fmt.Println("Matrix A | B")
	prtM(&AB)
	fmt.Println()

	A := AB
	A.M, A.N = 3, 3
	fmt.Println("A = ")
	prtM(&A)
	fmt.Println()

	B := AB
	B.M, B.N = 3, 3
	B.O_i, B.O_j = 3, 3
	fmt.Println("B = ")
	prtM(&B)
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

	blas.Dgemm(blas.NoTrans, blas.Trans, 1.0, &A, &B, 0.0, &C)
	fmt.Println("C = A*B= ")
	prtM(&C)
	fmt.Println()

	A.M, A.N = 2, 2
	B.M, B.N = 2, 2
	B.O_i, B.O_j = 1, 1
	fmt.Println("A = ")
	prtM(&A)
	fmt.Println()
	fmt.Println("B = ")
	prtM(&B)
	fmt.Println()
	blas.Dgemm(blas.NoTrans, blas.Trans, 1.0, &A, &B, 0.0, &C)
	fmt.Println("C = A*B=")
	prtM(&C)
	fmt.Println()
}

func prtM(A *blas.GeMatrix) {
	for i := 0; i < A.M; i++ {
		for j := 0; j < A.N; j++ {
			fmt.Printf("%6.1f", A.At(i, j))
		}
		fmt.Println()
	}
}
