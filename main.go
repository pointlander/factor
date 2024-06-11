// Copyright 2024 The Factor Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"github.com/pointlander/datum/iris"
)

func main() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	data := mat.NewDense(150, 4, nil)
	for i := range datum.Fisher {
		for j, value := range datum.Fisher[i].Measures {
			data.Set(i, j, value)
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
}
