// Copyright 2024 The Factor Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/kmeans"
	"github.com/pointlander/matrix"
)

func main() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	max := 0.0
	for i := range datum.Fisher {
		for _, value := range datum.Fisher[i].Measures {
			if value > max {
				max = value
			}
		}
	}

	data := mat.NewDense(150, 4, nil)
	orig := matrix.NewMatrix(4, 150)
	for i := range datum.Fisher {
		for j, value := range datum.Fisher[i].Measures {
			data.Set(i, j, value/max)
			orig.Data = append(orig.Data, float32(value/max))
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

	rawData := make([][]float64, sa.Rows)
	for i := 0; i < sa.Rows; i++ {
		for j := 0; j < sa.Cols; j++ {
			rawData[i] = append(rawData[i], float64(sa.Data[i*sa.Cols+j]))
		}
	}
	meta := matrix.NewMatrix(150, 150, make([]float32, 150*150)...)

	for i := 0; i < 100; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), rawData, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < 150; i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta.Data[i*150+j]++
				}
			}
		}
	}

	meta = matrix.SelfAttention(meta, meta, meta)

	x := make([][]float64, 150)
	for i := range x {
		x[i] = make([]float64, 150)
		for j := range x[i] {
			x[i][j] = float64(meta.Data[i*150+j])
		}
	}
	clusters, _, err := kmeans.Kmeans(1, x, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
	}
}
