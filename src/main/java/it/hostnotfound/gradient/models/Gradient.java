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

import java.util.ArrayList;
import java.util.Iterator;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import it.hostnotfound.gradient.models.minimizer.*;
import it.hostnotfound.gradient.models.mnist.*;

public class Gradient {
    private final static Logger LOG = LoggerFactory.getLogger(Gradient.class);

    /**
     * Contains the matrixes and methods to perform the Backpropagation algorithms
     */
    private NNComputer nnComputer;

    public Gradient(final NNComputer nnComputer) {
        this.nnComputer = nnComputer;
    }

    /**
     * Gradient must be instantiate through the Builder
     */
    private Gradient() {
    }

    /**
     * Builder design pattern with fluent interface
     */
    public static class Builder {

        /**
         * Activation matrix of the first layer, it includes the bias column made of 1.0
         */
        private NNMatrix A0;

        /**
         * Y are the ground truth, so the expected vector outputs.
         * It's needed if we are training or testing, not for prediction.
         */
        private NNMatrix Y;

        /**
         * Weights
         */
        private NNMatrix Weights[];

        /**
         * The number of input values
         */
        private int inputSize;

        /**
         * The number of items to classify so the number of dimensions of the output vector.
         */
        private int outputSize;

        /**
         * Constructor
         * 
         * @param inputSize number of elements for every input vector
         * @param outputSize number of elements of the output vector
         */
        public Builder(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
        }

        /**
         * Add weights matrixes
         * 
         * @param Weights array of weights matrixes
         * @return this
         */
        public Builder withWeights(final NNMatrix[] Weights) {
            this.Weights = Weights;

            return this;
        }

        /**
         * Load the train set either from matrix file of using MNIST archive files
         * 
         * @return this
         * @throws Exception on errors
         */
        public Builder loadTrainDataSet() throws Exception {
            String a0FileName = "a0.matrix";
            String yFileName  = "y.matrix";

            if (Files.exists(Paths.get(a0FileName)) && Files.exists(Paths.get(yFileName))) {
                A0 = NNMatrix.loadFromFile(a0FileName);
                LOG.info("A0 train set matrix loaded");

                if (A0.numCols() != (inputSize + 1)) {
                    throw new Exception("The A0 matrix is not compatible with the given inputSize");
                }

                Y = NNMatrix.loadFromFile(yFileName);
                LOG.info("Y train set matrix loaded");

                if (Y.numCols() != outputSize) {
                    throw new Exception("The Y matrix is not compatible with the given outputSize");
                }
            } else {
                ArrayList<MNISTItem> dataSet = new ArrayList<MNISTItem>();
                try {
                    MNISTReader reader = new MNISTReader();
                    dataSet = reader.readTrainSet();
                } catch(Exception e) {
                    throw new Exception("Failed to load the train dataset");
                }

                if (dataSet.get(0).size() != inputSize) {
                    throw new Exception("The given input size doesn't match with the input size from the dataset");
                }

                trainMatrixesFromDataset(dataSet);
            }

            return this;
        }

        /**
         * Build the train matrixes A0 and Y from MNIST dataset
         */
        protected void trainMatrixesFromDataset(ArrayList<MNISTItem> dataSet) {
            A0 = new NNMatrix(dataSet.size(), inputSize + 1);
            Y = new NNMatrix(dataSet.size(), outputSize);

            // set values for A0 and Y
            Iterator<MNISTItem> rows = dataSet.iterator();
            int i = 0;
            while (rows.hasNext()) {
                MNISTItem e = rows.next();
                double[] r = e.getDoublePixels();
                // A0[i,0] = 1.0
                A0.set(i, 0, 1.0);
                // add input vectors [<--- as a rows --->]
                for (int j = 0; j < e.size(); j++) {
                    A0.set(i, 1 + j, r[j]);
                }
                // add expected output vectors [<--- as a rows --->]
                for (int j = 0; j < outputSize; j++) {
                    if (e.getLabel() == j) {
                        Y.set(i, j, 1.0);
                    } else {
                        Y.set(i, j, 0.0);
                    }
                }
                i++;
            }
        }

        /**
         * Build the Gradient object
         * 
         * @return
         * @throws Exception on error
         */
        public Gradient build() throws Exception {
            if (Weights == null) {
                throw new Exception("Cannot build Gradient object without Weights");
            }
            if ((A0 != null) && (Y != null)) {
                NNComputer nnComputer = new NNComputer(A0, Y, Weights);
                Gradient.validate(nnComputer);
                return new Gradient(nnComputer);
            }
            return new Gradient(new NNComputer(Weights));
        }

    }
    
    /**
     * load weights matrixes from files
     * 
     * @param numWeightsMatrixes total number of matrixes to load
     * @return matrixes
     * @throws Exception thrown if file was not found or an I/O error occurred
     */
    public static NNMatrix[] loadWeights(final int numWeightsMatrixes) throws Exception {
        NNMatrix W[] = new NNMatrix[numWeightsMatrixes];
        String fileName;
        for (int i = 0; i < numWeightsMatrixes; i++) {
            fileName = "weights_" + i + ".matrix";
            if (!Files.exists(Paths.get(fileName))) {
                throw new FileNotFoundException("Failed to load the weight file: " + fileName);
            }
            W[i] = NNMatrix.loadFromFile(fileName);
            LOG.info("Weight matrix " + i + " loaded");
        }
        return W;
    }

    /**
     * Save the status of the training session
     * 
     * @throws IOException
     */
    protected void save() throws IOException {
        validate(nnComputer);
        NNMatrix A0 = nnComputer.getA0();
        NNMatrix Y = nnComputer.getY();
        NNMatrix W[] = nnComputer.getWeights();

        // save first activation layer and output
        A0.saveToFileBinary("a0.matrix");
        Y.saveToFileBinary("y.matrix");

        // save weights
        for (int i = 0; i < W.length; i++) {
            W[i].saveToFileBinary("weights_" + i + ".matrix");
        }
    }

    /**
     * Validate the weighs matrixes for every layer
     * 
     * @throws Error
     */
    public static void validate(NNComputer nnComputer) throws Error {
        NNMatrix A0 = nnComputer.getA0();
        NNMatrix Y = nnComputer.getY();
        NNMatrix Weights[] = nnComputer.getWeights();

        //final NNMatrix Weights[] = nnComputer.getWeights();
        int prevRows = A0.numCols() - 1; // since A0 is transpose(X with bias)
        int rows = 0;
        int cols = 0;
        LOG.trace("X input matrix " + (A0.numCols() - 1) + "x" + A0.numRows() + "");
        LOG.trace("A0 (transpose input vectors plus the bias element): " + A0.numRows() + "x" + A0.numCols());
        for (int i = 0; i < Weights.length; i++) {
            if (Weights[i] == null) {
                throw new Error("weight matrix for the layer " + (i) + " is not defined");
            }
            rows = Weights[i].numRows();
            cols = Weights[i].numCols();
            LOG.trace("layer " + (i) + ", weight matrix " + rows + "x" + cols + "");
            if ((prevRows + 1) != cols) {
                if (i == 0) {
                    throw new Error("weight matrix on layer " + (i) + " has wrong size (" + rows + "x" + cols
                            + "), the input vector has " + prevRows + " columns");
                } else {
                    throw new Error("weight matrix on layer " + (i) + " has wrong size (" + rows + "x" + cols
                            + "), the previous matrix had " + prevRows + " columns");
                }
            }
            prevRows = rows;
        }
        // checking the matrix between last layer -> output layer
        if (prevRows != Y.numCols()) {
            throw new Error("weight matrix on layer " + (Weights.length - 1) + " has wrong size (" + rows + "x" + cols
                    + "), the output vector has " + Y.numCols() + " elements");
        }
        LOG.trace("Y output matrix " + Y.numRows() + "x" + Y.numCols() + "");
    }

    /**
     * Train neural network using conjugated gradient descent
     * 
     * @param lambda
     * @param maxEpoch
     */
    public void trainFmincg(final double lambda, final int maxEpoch) {
        Fmincg minimizer = new Fmincg(nnComputer, maxEpoch);
        train(minimizer, lambda);
    }

    /**
     * Train neural network using plain batch gradient descent
     * 
     * @param alpha
     * @param lambda
     * @param maxEpoch
     */
    public void trainNaive(final double alpha, final double lambda, final int maxEpoch) {
        Naive minimizer = new Naive(nnComputer, maxEpoch, alpha);
        train(minimizer, lambda);
    }

    /**
     * Train neural network with the given minimizer
     * 
     * @param minimizer
     * @param lambda
     */
    public void train(Minimizer minimizer, final double lambda) {
        validate(nnComputer);
        nnComputer.setLambda(lambda);
        minimizer.minimize();
        try {
            save();
        } catch (IOException e) {
            LOG.error("Failed to save matrixes");
        }
    }

    /**
     * Predict digit from the given image
     * 
     * @param image
     */
    public int predict(final MNISTItem image) {
        final NNMatrix x = new NNMatrix(1, image.size(), image.getDoublePixels());

        final NNMatrix p = nnComputer.predict(x);
        //p.print();
        int predicted=0;
        for(int i=0; i < p.getNumElements(); i++) {
            if (p.get(predicted) < p.get(i)) {
                predicted=i;
            }
        }
        return predicted;
    }

}
