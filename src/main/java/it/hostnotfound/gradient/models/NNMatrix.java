/**
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, Samuele Catuzzi
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package it.hostnotfound.gradient.models;

import java.io.IOException;
import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.data.Matrix;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleBase;
import org.ejml.simple.SimpleMatrix;
import org.ejml.dense.row.CommonOps_DDRM;

/**
 * Matrix of elements of type double
 */
public class NNMatrix extends SimpleBase<NNMatrix> {

    static final long serialVersionUID = 2342556643L;

    public NNMatrix(Matrix orig) {
        // this constructor is needed in order to satisfy SimpleBase.wrapMatrix()
        setMatrix(orig.copy());
    }

    public NNMatrix(int numRows, int numCols) {
        setMatrix(new DMatrixRMaj(numRows, numCols));
    }

    public NNMatrix(double data[][]) {
        setMatrix(new DMatrixRMaj(data));
    }

    public NNMatrix(double data[]) {
        setMatrix(new DMatrixRMaj(data));
    }

    public NNMatrix(int numRows, int numCols, double data[]) {
        setMatrix(new DMatrixRMaj(numRows, numCols, true, data));
    }

    public double[] getData() {
        return ((DMatrixRMaj) getMatrix()).getData().clone();
    }

    public void setData( double[] data ) {
		((DMatrixRMaj) getMatrix()).setData(data.clone());
	}

    public NNMatrix copy() {
        return new NNMatrix(numRows(), numCols(), getData());
    }

    public static NNMatrix loadFromFile(String fileName) throws IOException {
        SimpleMatrix mat = SimpleMatrix.loadBinary(fileName);
        return new NNMatrix(mat.numRows(), mat.numCols(), ((DMatrixRMaj) mat.getMatrix()).getData());
    }

    public static NNMatrix randomizedMatrix(int numRows, int numCols, double min, double max) {
        NNMatrix R = new NNMatrix(numRows, numCols);
        R.randomizeElements(min, max);
        return R;
    }

    public void randomizeElements(double min, double max) {
        final Random rand = new Random();
        for (int i = 0; i < numRows(); i++) {
            for (int j = 0; j < numCols(); j++) {
                set(i, j, rand.nextDouble() * (max - min) + min);
            }
        }
    }

    /**
     * Apply sigmoid function on every element of the matrix
     * 
     * @return return the sigmoid matrix of the matrix
     */
    public NNMatrix sigmoid() {
        final double data[] = getData();
        double x;
        for (int i = 0; i < data.length; i++) {
            x = data[i];
            // sigmoid function
            data[i] = 1 / (1 + Math.exp(-x));
        }

        return new NNMatrix(numRows(), numCols(), data);
    }

    /**
     * Apply sigmoid derivative on every element of the matrix
     * 
     * @return return the sigmoid derivative matrix of the matrix
     */
    public NNMatrix sigmoidDerivative() {
        final double data[] = getData();
        double s, x;
        for (int i = 0; i < data.length; i++) {
            x = data[i];
            // sigmoid function
            s = 1 / (1 + Math.exp(-x));
            // sigmoid derivative
            data[i] = s * (1 - s);
        }

        return new NNMatrix(numRows(), numCols(), data);
    }

    /**
     * Prepend the matrix with an extra column made by elements with the give value
     * 
     * @param value the value of every element of the new column
     * @return a copy of the matrix with the extra column
     */
    public NNMatrix prependColumnValue(double value) {
        final NNMatrix A = new NNMatrix(numRows(), numCols() + 1);
        A.insertIntoThis(0, 1, this);
        final int size = A.numRows();
        // todo: check if correct
        for (int i = 0; i < size; i++) {
            A.setColumn(0, i, value);
        }
        return A;
    }

    /**
     * Set all elements of the give column to the specified value
     * 
     * @param column the index of the column
     * @param value the value to be set for every element of the column
     * @return a copy of the matrix with the give column made of value
     */
    public NNMatrix setColumnValue(int column, double value) {
        final NNMatrix A = copy();
        final int size = numRows();
        for (int i = 0; i < size; i++) {
            A.set(i, column, value);
        }
        return A;
    }

    /**
     * Calculate: this matrix multiplied for the B matrix transposed
     * 
     * @param B matrix
     * @return A * (B') , where A is this matrix
     */
    public NNMatrix multTransposedB(NNMatrix B) {
        NNMatrix C = new NNMatrix(numRows(), B.numCols());
        CommonOps_DDRM.multTransB(((DMatrixRMaj) getMatrix()), (DMatrixRMaj) B.getMatrix(), (DMatrixRMaj) C.getMatrix());
        return C;
    }

    /**
     * Calculate: this matrix transposed and after, multiplied for the B matrix
     * 
     * @param B matrix
     * @return (A') * B  , where A is this matrix
     */
    public NNMatrix transposedMultB(NNMatrix B) {
        NNMatrix C = new NNMatrix(numCols(), B.numCols());
        CommonOps_DDRM.multTransA(((DMatrixRMaj) getMatrix()), (DMatrixRMaj) B.getMatrix(), (DMatrixRMaj) C.getMatrix());
        return C;
    }

    /**
     * Flat the given matrixes into a single vector combining all their elements
     * 
     * @param matrixes array of matrixes to be flatten into a single row
     * @return single row matrix made of all the elements of the given matrixes
     */
    public static NNMatrix matrixesToVector(final NNMatrix[] matrixes) {
        NNMatrix R, M;
        R = matrixes[0].copy();
        R.reshape(1, matrixes[0].getNumElements());
        for(int i=1; i < matrixes.length; i++) {
            // reshape the matrix as single row
            M = matrixes[i].copy();
            M.reshape(1, matrixes[i].getNumElements());
            // concatenate
            R = R.combine(0, R.getNumElements(), M);
        }
        return R;
    }

    @Override
    protected NNMatrix createMatrix(int numRows, int numCols, MatrixType type) {
        return new NNMatrix(numRows, numCols);
    }

    @Override
    protected NNMatrix wrapMatrix(Matrix orig) {
        return new NNMatrix(orig);
    }
}
