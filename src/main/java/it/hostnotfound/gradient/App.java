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

package it.hostnotfound.gradient;

import java.io.File;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import it.hostnotfound.gradient.models.Gradient;
import it.hostnotfound.gradient.models.NNMatrix;
import it.hostnotfound.gradient.models.mnist.*;

/**
 * Gradient descent over MNIST database
 */
public class App {

    private final static int INPUT_SIZE = 28 * 28;
    private final static int OUTPUT_SIZE = 10;
    private final static int HIDDEN_LAYER_SIZE = 25;

    protected Gradient grad;

    public static void main(final String[] args) {
        try {
            App app = new App(args);
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.exit(1);
        }
    }

    public App(final String[] args) throws Exception {

        String op = "";
        int predictItem = 0;

        System.out.println("Gradient descent using MNIST database");

        // very simple argument parser
        if (args.length < 1
                || !args[0].equalsIgnoreCase("train") && !args[0].equalsIgnoreCase("predict")
                        && !args[0].equalsIgnoreCase("test")
                || args[0].equalsIgnoreCase("predict") && args.length != 2) {
            throw new IllegalArgumentException("invalid arguments");
        }
        if (args[0].equalsIgnoreCase("predict")) {
            try {
                predictItem = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("provided argument '" + args[1] + "' must be an integer.");
            }
        }
        op = args[0].toLowerCase();

        init(op, predictItem);
    }

    /**
     * Initialize and process
     * 
     * @param op          can be one of "train", "predict"
     * @param predictItem
     * @throws Exception
     */
    protected void init(String op, int predictItem) throws Exception {
        // load weights from the disk or initialize new ones
        NNMatrix weights[] = null;
        try {
            weights = Gradient.loadWeights(2);
        } catch (final Exception e) {
            System.out.println("weights files not found, initializing new ones");

            weights = new NNMatrix[2];

            // initialize weight (including the bias) for the X -> hidden layer
            weights[0] = NNMatrix.randomizedMatrix(HIDDEN_LAYER_SIZE, INPUT_SIZE + 1, -1, 1);

            // initialize weights for hidden layer + bias -> Y (Y has 10 possible values, it
            // is a simple vector made of 10 elements)
            weights[1] = NNMatrix.randomizedMatrix(OUTPUT_SIZE, HIDDEN_LAYER_SIZE + 1, -1, 1);
        }

        process(op, predictItem, weights);
    }

    protected void process(String op, int predictItem, NNMatrix[] weights) throws Exception {
        // start processing
        if (op.equalsIgnoreCase("train")) {
            train(weights);

        } else if (op.equalsIgnoreCase("predict") || op.equalsIgnoreCase("test")) {
            evaluate(op, predictItem, weights);
        }
    }

    /**
     * Train neural network
     * 
     * @param weights the weights matrixes of the network
     * @throws Exception
     */
    protected void train(NNMatrix[] weights) throws Exception {
        final double lambda = 1.0;
        final double alpha = 0.1;

        grad = new Gradient.Builder(INPUT_SIZE, OUTPUT_SIZE).withWeights(weights).loadTrainDataSet().build();

        grad.trainNaive(alpha, lambda, 10);
        // grad.trainFmincg(lambda, 100);
    }

    /**
     * Predict one of the test set images or test all the test set images and report
     * 
     * @param op          can be "predict" or "test"
     * @param predictItem the index position of the image to test from the test set
     *                    image collection
     * @param weights     the weights matrixes of the network
     * @throws Exception
     */
    protected void evaluate(final String op, final int predictItem, NNMatrix[] weights) throws Exception {

        grad = new Gradient.Builder(INPUT_SIZE, OUTPUT_SIZE).withWeights(weights).build();

        final MNISTReader reader = new MNISTReader();

        ArrayList<MNISTItem> testSet = reader.readTestSet();
        MNISTItem testImage = null;

        if (op.equalsIgnoreCase("predict")) {
            if ((predictItem < 0) || (predictItem >= testSet.size())) {
                throw new IllegalArgumentException(
                        "Error, the id item from the test set to be predicted can only be a value in the range 0-"
                                + (testSet.size() - 1));
            }
            // save the predictItem image on file for a visual inspection
            final File outputFile = new File("digit.png");
            ImageIO.write(testSet.get(predictItem).getImage(), "png", outputFile);

            testImage = testSet.get(predictItem);
            System.out.println("expected output: " + testImage.getLabel());
            System.out.println("predicted value: " + grad.predict(testImage));
        } else {
            // predict digits from the whole test set images
            int correct = 0;
            for (int i = 0; i < testSet.size(); i++) {
                testImage = testSet.get(i);
                if (testImage.getLabel() == grad.predict(testImage)) {
                    correct++;
                } else {
                    // System.out.print(i + " ,");
                }
            }
            System.out.println("");
            System.out.println("# of correctly classified images: " + correct + " over " + testSet.size()
                    + " taken from the test set");
        }
    }

}
