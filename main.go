package main

import (
	"airfoil_API/M_Matrix"
	"airfoil_API/P_Matrix"
	"airfoil_API/Parsec"
	"encoding/csv"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type Airfoil struct {
	D2Yl float64 `json:"d2Yl"`
	Y_TE float64 `json:"y_TE"`
	A_TE float64 `json:"a_TE"`
	AOA  float64 `json:"aoa"`
}

func APIplotAirfoil(c *gin.Context) {
	var airfoil Airfoil

	if err := c.BindJSON(&airfoil); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Plot the airfoil
	d2Yl := airfoil.D2Yl
	y_TE := airfoil.Y_TE
	a_TE := airfoil.A_TE
	AOA := airfoil.AOA

	generateAirfoil(0.04, 0.3, 0.3, 0.11, -0.8, -0.06, d2Yl, 0.000000, y_TE, a_TE, 10, AOA, "airfoil")

	// Save the plot to a file
	// Return status 200 OK
	c.JSON(http.StatusOK, gin.H{"message": "Plot generated and saved successfully"})
}

func main() {
	router := gin.Default()
	router.POST("/plotAirfoil", APIplotAirfoil)
	router.Run(":8080")
}

func generateAirfoil(R_LE, Xu, Xl, Yu, d2Yu, Yl, d2Yl, del_Y_TE, Y_TE, a_TE, b_TE, AOA float64, fileName string) {
	M := M_Matrix.CreateM(Xu, Xl)
	P := P_Matrix.CreateP(R_LE, Yu, d2Yu, Yl, d2Yl, del_Y_TE, Y_TE, a_TE, b_TE)
	m := M_Matrix.GetInverse(M)
	solution := P_Matrix.Multiply(m, P)
	yUpper, yLower, X := Parsec.GenerateAirfoilUpper(solution)
	plotAirfoil(X, yUpper, yLower, true, AOA, fileName)
}

func plotAirfoil(X, YUpper, YLower []float64, isValid bool, AOA float64, saveName string) {

	ptsUpper := make(plotter.XYs, len(X))
	ptsLower := make(plotter.XYs, len(X))

	for i := range X {
		ptsUpper[i].X = X[i] / 2
		ptsUpper[i].Y = YUpper[i] / 2

		ptsLower[i].X = X[i] / 2
		ptsLower[i].Y = YLower[i] / 2
	}

	p := plot.New()

	p.Title.Text = "Y_Upper and Y_Lower Curves"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err := plotutil.AddLinePoints(p,
		"Y_Upper", ptsUpper,
		"Y_Lower", ptsLower,
	)
	if err != nil {
		log.Fatalf("plotting failed: %v", err)
	}

	dir, err := os.Getwd()
	if err != nil {
		fmt.Println("Error getting current directory: ", err)
		return
	}

	if isValid {

		savePath := dir + "\\outputs\\" + saveName + ".png"
		_ = p.Save(32*vg.Inch, 6*vg.Inch, savePath)
		saveToCSV(X, YUpper, YLower, saveName)
	}

}

func saveToCSV(X, YUpper, YLower []float64, saveName string) {
	// Get current Directory
	dir, err := os.Getwd()
	if err != nil {
		fmt.Println("Error getting current directory: ", err)
		return
	}
	// Create a csv file
	file, err := os.Create(dir + "\\outputs\\" + saveName + ".csv")
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Generate the arrays required to print
	// X repeats twice
	// Y_Upper and Y_Lower are appended to each other
	outputX := make([]float64, (len(X)*2)-1)
	outputY := make([]float64, (len(YUpper) + len(YLower) - 1))
	outputZ := make([]float64, (len(X)*2)-1)
	max := len(X)

	for i := range X {
		outputX[i] = X[i] / 2
		outputX[i+len(X)-1] = X[max-i-1] / 2
		outputY[i] = YUpper[i] / 2
		outputY[i+len(X)-1] = YLower[max-i-1] / 2
		outputZ[i] = 0
		outputZ[i+len(X)-1] = 0
	}

	// Write X, Y_Upper, and Y_Lower to the csv file
	for i := range outputX {
		if i == len(outputX)-1 {
			break
		}
		err := writer.Write([]string{fmt.Sprintf("%f", outputX[i]), fmt.Sprintf("%f", outputY[i]), fmt.Sprintf("%f", outputZ[i])})
		if err != nil {
			log.Fatalf("failed writing to file: %s", err)
		}
	}
}
