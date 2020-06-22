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

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NNComputer {
    final static Logger LOG = LoggerFactory.getLogger(NNComputer.class);

    /**
     * Y is the ground truth, it is the vector of expected outputs
     */
    protected NNMatrix Y;

    /**
     * Weights contains a list of matrixes representing the weights and the biases
     * of each layer of the neural network.
     * Where Weights[0] is the matrix from the input X to the first hidden layer.
     */
    protected NNMatrix[] Weights;

    /**
     * Activations contains a list of matrix representing the activation, those matrixes have
     * a column of ones (1.0) for the biases.
     * Where Activations[0] is the input values with the column of ones for the bias.
     */
    protected NNMatrix[] Activations;

    /**
     * cost
     */
    protected double J = 0;

    /**
     * regularization parameter
     */
    protected double lambda = 1.0;

    /**
     * H is last activation layer, the predicted vector outputs 
     */
    private NNMatrix H;
    private NNMatrix Z0;
    private NNMatrix[] gradients;

    public NNComputer(NNMatrix[] Weights) {
        Activations = new NNMatrix[Weights.length];
        Arrays.fill(Activations, null);
        this.Weights = Weights;
    }

    public NNComputer(NNMatrix A0, NNMatrix Y, NNMatrix[] Weights) {
        this(A0, Y, Weights, 1.0);
    }

    public NNComputer(NNMatrix A0, NNMatrix Y, NNMatrix[] Weights, double lambda) {
        Activations = new NNMatrix[Weights.length];
        Arrays.fill(Activations, null);
        Activations[0] = A0;
        this.Y = Y;
        this.Weights = copy(Weights);
        this.lambda = lambda;
    }

    public NNComputer(NNMatrix A0, NNMatrix Y, NNMatrix WeightsAsVector, int[][] weightsDim) throws Exception {
        this(A0, Y, WeightsAsVector, weightsDim, 1.0);
    }

    public NNComputer(NNMatrix A0, NNMatrix Y, NNMatrix WeightsAsVector, int[][] weightsDim, double lambda) throws Exception {
        int size=0;
        for(int i=0; i < weightsDim.length; i++) {
            if (weightsDim[i].length != 2) {
                throw new Exception("The dimensions for the weights matrix are not two");
            }
            size = weightsDim[i][0] * weightsDim[i][1];
            // reconstruct weights matrixes
            NNMatrix mat = WeightsAsVector.extractMatrix(0, 1, 0, size);
            mat.reshape(weightsDim[i][0], weightsDim[i][1] + 1);
            Weights[i] = mat;
        }
        Activations = new NNMatrix[Weights.length];
        Arrays.fill(Activations, null);
        Activations[0] = A0.copy();
        this.Y = Y;
        this.lambda = lambda;
    }

    public NNComputer(NNComputer nnComputer) {
        this(nnComputer.Activations[0], nnComputer.Y, nnComputer.Weights, nnComputer.lambda);
    }

    public NNMatrix getA0() {
        return Activations[0].copy();
    }

    public void setA0(NNMatrix A0) {
        Activations[0] = A0.copy();
    }

    public NNMatrix getY() {
        return Y.copy();
    }

    public void setY(NNMatrix Y) {
        this.Y = Y.copy();
    }

    public NNMatrix[] getWeights() {
        return copy(Weights);
    }

    public void setWeights(NNMatrix[] weights) {
        Weights = copy(weights);
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double getCost() {
        return J;
    }

    public NNMatrix[] getGradients() {
        return copy(gradients);
    }

    private NNMatrix[] copy(NNMatrix[] mat) {
        NNMatrix C[] = new NNMatrix[mat.length];
        for(int wi=0; wi < mat.length; wi++) {
            C[wi] = mat[wi].copy();
        }
        return C;
    }
    
    /**
     * Perform computation
     */
    public void compute() {
        forwardPass();
        backwardPass();
        costFunction();
    }

    /**
     * Predict
     * 
     * @param input given item
     * @return predicted output vector
     */
    public NNMatrix predict(NNMatrix input) {
        Activations[0] = input.prependColumnValue(1.0);
        forwardPass();
        return H.transpose();
    }

    protected void forwardPass() {
        LOG.trace("W0 matrix "+Weights[0].numRows()+"x"+Weights[0].numCols()+"");
        // perform Z0=A0*W0'
        Z0 = Activations[0].multTransposedB(Weights[0]);
        LOG.trace("Z0 matrix "+Z0.numRows()+"x"+Z0.numCols()+"");

        NNMatrix H1 = Z0.sigmoid();
        LOG.trace("H1 matrix "+H1.numRows()+"x"+H1.numCols()+"");
        
        // prepend a column of 1.0 to the unbiased matrix to get the biased activation matrix A1
        Activations[1] = H1.prependColumnValue(1.0);

        LOG.trace("W1 matrix "+Weights[1].numRows()+"x"+Weights[1].numCols()+"");
        // perform Z1=A1*W1'
        NNMatrix Z1 = Activations[1].multTransposedB(Weights[1]);
        LOG.trace("Z1 matrix "+Z1.numRows()+"x"+Z1.numCols()+"");

        // H=2H which is the final prediction
        H = Z1.sigmoid();
        LOG.trace("H matrix "+H.numRows()+"x"+H.numCols()+"");
    }

    protected void backwardPass() {
        gradients = new NNMatrix[2];

        // δ(L) = a(L) − y(t)
        NNMatrix D2_ = H.minus(Y);

        NNMatrix sigGrad = Z0.sigmoidDerivative().prependColumnValue(1.0);

        NNMatrix T_ = D2_.mult(Weights[1]).elementMult(sigGrad);

        // δ(l) = ( Θ(l)' δ(l+1) ) .∗ a(l) .∗ ( 1 − a(l) )
        // remove the first column
        NNMatrix D1_ = T_.extractMatrix(0, T_.numRows(), 1, T_.numCols());

        // regularization part -------
        double m = Activations[0].numRows();

        // (1.0 / m) * { (D1_') * A0 }
        NNMatrix WGrad1_ = D1_.transposedMultB(Activations[0]).scale(1.0 / m);
        NNMatrix WGrad2_ = D2_.transposedMultB(Activations[1]).scale(1.0 / m);

        // exclude theta0 from regularization
        T_ = WGrad1_.setColumnValue(0, 0.0).scale(lambda / m);
        // regularize
        gradients[0] = WGrad1_.plus(T_);

        // same for the other 
        T_ = WGrad2_.setColumnValue(0, 0.0).scale(lambda / m);

        gradients[1] = WGrad2_.plus(T_);
    }

    /**
     * Calculate the cost using cost function with regularization
     */
    protected void costFunction() {
        double m=Activations[0].numCols();

        // J = (-1/m) * SumAllElements( Y*log(H) + (1 - Y)*log(1 - H) )
        NNMatrix T1_ = Y.elementMult(H.elementLog());
        NNMatrix T2_ = Y.negative().plus(1.0).elementMult(H.negative().plus(1.0).elementLog());
        J = (-1/m) * T1_.plus(T2_).elementSum();

        // regularization part -------
        NNMatrix W0_ = Weights[0];
        NNMatrix W1_ = Weights[1];

        double regW0 = W0_.extractMatrix(0, W0_.numRows(), 1, W0_.numCols()).elementPower(2.0).elementSum();
        double regW1 = W1_.extractMatrix(0, W1_.numRows(), 1, W1_.numCols()).elementPower(2.0).elementSum();

        J = J + (lambda/(2*m)) * (regW0 + regW1);
    }

}
