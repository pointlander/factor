// Copyright 2024 The Factor Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/matrix"
)

func main() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	data := mat.NewDense(150, 4, nil)
	orig := matrix.NewMatrix(4, 150)
	for i := range datum.Fisher {
		for j, value := range datum.Fisher[i].Measures {
			data.Set(i, j, value)
			orig.Data = append(orig.Data, float32(value))
		}
	}

	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		panic("pca error")
	}
	fmt.Printf("variances = %.4f\n\n", pc.VarsTo(nil))

	k := 4
	var proj mat.Dense
	var vec mat.Dense
	pc.VectorsTo(&vec)
	proj.Mul(data, vec.Slice(0, 4, 0, k))

	fmt.Printf("proj = %.4f\n", mat.Formatted(&proj, mat.Prefix("       ")))

	in := matrix.NewMatrix(4, 150)
	for i := 0; i < 150; i++ {
		for j := 0; j < 4; j++ {
			in.Data = append(in.Data, float32(proj.At(i, j)))
		}
	}
	sa := matrix.SelfAttention(in, in, in)
	for i := range datum.Fisher {
		for j := 0; j < 4; j++ {
			fmt.Printf("%f ", sa.Data[i*4+j])
		}
		fmt.Printf("%s\n", datum.Fisher[i].Label)
	}
}
