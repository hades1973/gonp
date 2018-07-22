package main

import (
	"fmt"
	"gonp/blas"
)

var data []float64 = []float64{
	1, 1, 1, 6,
	0, 4, -1, 5,
	2, -2, 1, 1,
}

func prtData() {
	var g = func(i, j int) float64 {
		return data[i*4+j]
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			fmt.Printf("%6.1f", g(i, j))

		}
		fmt.Println()
	}
}

func main() {
	fmt.Println("vim-go")
	fmt.Println("Raw Majorx:")
	prtData()

	r_1 := blas.Vector{
		N:    4,
		Inc:  1,
		Off:  0,
		Base: data,
	}
	r_2 := blas.Vector{
		N:    4,
		Inc:  1,
		Off:  4,
		Base: data,
	}
	r_3 := blas.Vector{
		N:    4,
		Inc:  1,
		Off:  8,
		Base: data,
	}

	fmt.Println("(-2) x r_1 + r_3 -> r_3")
	alpha := -2.0
	blas.Daxpy(alpha, r_1, r_3)
	prtData()

	fmt.Println("r_2 + r_3 -> r_3")
	alpha = 1.0
	blas.Daxpy(alpha, r_2, r_3)
	prtData()

	fmt.Println("r_2 <-> r_3")
	blas.Dswap(r_2, r_3)
	prtData()

	fmt.Println("r_2 \\dot r_3")
	dot := blas.Ddot(r_2, r_3)
	fmt.Println(dot)
}
