package main

import (
	"fmt"
	"gonp/blas"
)

func main() {
	// 此种方式存储，不能使用转置参数
	// |     A     |  x  | b |
	// -----------------------
	// |10, -7, 0, |  0, | 0,|
	// |-3, +2, 6, | -1, | 0,|
	// | 5, -1, 5, |  1, | 0,|
	// -----------------------
	var Axb = blas.GeMatrix{
		M: 3, N: 5, LD: 5,
		OffRow: 0, OffCol: 0,
		Base: []float64{
			10, -7, 0, 0, 0,
			-3, +2, 6, -1, 0,
			+5, -1, 5, 1, 0,
		},
	}

	A := Axb
	A.N = 3
	fmt.Println("Matrix Axb")
	prtM(&Axb)
	fmt.Println()
	fmt.Println("Matrix A:")
	prtM(&A)

	x := Axb.Col(3)
	b := Axb.Col(4)
	blas.Dgemv(blas.NoTrans, 1.0, &A, x, 0, b)

	fmt.Println("After call Degmv, Matrix Axb")
	prtM(&Axb)

	// |   A          | x   | b |
	// -----------------------
	//     B
	// =========
	// |10, -7,||   0,   0, | 7,|
	//            ===========
	// |-3, +2,|| ||6,  -1, || 4,|
	// =========  ||
	// | 5, -1,   ||5,   1, || 6,|
	//            ===========
	//                C
	// -----------------------
	B := Axb
	B.M, B.N = 2, 2
	C := Axb
	C.M, C.N = 2, 2
	C.OffRow, C.OffCol = 1, 2
	D := blas.GeMatrix{
		M: 2, N: 2, LD: 2,
		OffRow: 0, OffCol: 0,
		Base: []float64{
			0., 0.,
			0., 0.,
		},
	}
	blas.Dgemm(blas.NoTrans, blas.Trans, 1.0, &B, &C, 0.0, &D)
	fmt.Println("--------------")
	prtM(&B)
	fmt.Println("--------------")
	prtM(&C)
	fmt.Println("--------------")
	prtM(&D)
	fmt.Println("--------------")

}

func prtM(A *blas.GeMatrix) {
	for i := 0; i < A.M; i++ {
		for j := 0; j < A.N; j++ {
			fmt.Printf("%6.1f", A.At(i, j))
		}
		fmt.Println()
	}
}
