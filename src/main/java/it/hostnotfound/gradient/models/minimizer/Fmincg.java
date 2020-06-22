package it.hostnotfound.gradient.models.minimizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import org.ejml.data.DMatrixD1;
import org.ejml.dense.row.CommonOps_DDRM;

import it.hostnotfound.gradient.models.NNComputer;
import it.hostnotfound.gradient.models.NNMatrix;

/**
 * Created by Samuele Catuzzi on 2020-06-13
 * 
 * copied and adapted from:
 * https://github.com/xmichaelx/computational_intelligence_project-neural_net/blob/master/src/main/java/com/sybillatechnologies/research/Fmincg.java
 * 
 * which in turn was copied from:
 * https://github.com/thomasjungblut/thomasjungblut-common/blob/master/src/de/jungblut/math/minimize
 * 
 * which is a translation in Java of an existent Matlab/Octave code from: 
 * Andrew Ng Machine Learning Course on Coursera (which I've attend as well)
 * 
 * ...and the following was the original copyright from the Matlab/Octave code:
 * 
 * % Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
 * %
 * %
 * % (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
 * % 
 * % Permission is granted for anyone to copy, use, or modify these
 * % programs and accompanying documents for purposes of research or
 * % education, provided this copyright notice is retained, and note is
 * % made of any changes that have been made.
 * % 
 * % These programs and documents are distributed without any warranty,
 * % express or implied.  As the programs were written for research
 * % purposes only, they have not been tested to the degree that would be
 * % advisable in any important application.  All use of these programs is
 * % entirely at the user's own risk.
 */
public class Fmincg implements Minimizer {
    private final static Logger LOG = LoggerFactory.getLogger(Fmincg.class);

    private NNComputer nnComputer;
    private int maxEpoch;

    // extrapolate maximum 3 times the current bracket.
    // this can be set higher for bigger extrapolations
    public static double EXT = 3.0;
    // a bunch of constants for line searches
    // RHO and SIG are the constants in the Wolfe-Powell conditions
    private static final double RHO = 0.01;
    private static final double SIG = 0.5;
    // don't reevaluate within 0.1 of the limit of the current bracket
    private static final double INT = 0.1;
    // max 20 function evaluations per line search
    private static final int MAX = 20;
    // maximum allowed slope ratio
    private static final int RATIO = 100;

    public Fmincg(final NNComputer nnComputer, int maxEpoch) {
        this.nnComputer = nnComputer;
        this.maxEpoch = maxEpoch;
    }

    protected NNMatrix[] vectorToWeights(NNMatrix vector) {
        // uses the known size of the weight matrixes to reconstruct their data and shape
        NNMatrix w[] = nnComputer.getWeights();
        double vecData[] = vector.getData();
        int start = 0;
        int numElem = 0;
        for(int i=0; i< w.length; i++) {
            numElem=w[i].getNumElements();
            w[i].setData(Arrays.copyOfRange(vecData, start, start + numElem));
            start += numElem;
        }
        return w;
    }

    @Override
    public void minimize() {
        int length = maxEpoch;
        NNMatrix inputVector = NNMatrix.matrixesToVector(nnComputer.getWeights());

        int M = 0;
        int i = 0; // zero the run length counter
        int red = 1; // starting point
        int ls_failed = 0; // no previous line search has failed
        // CostGradientTuple evaluateCost = f.getCost(inputVector);
        // double f1 = evaluateCost.getJ();
        // NNMatrix df1 = evaluateCost.getGrad();
        NNComputer evaluateCost = new NNComputer(nnComputer);
        evaluateCost.compute();
        double f1 = evaluateCost.getCost();
        NNMatrix df1 = NNMatrix.matrixesToVector(evaluateCost.getGradients());
        LOG.debug(String.format("epoch #%02d: J = %f ", i, f1));

        i += (length < 0 ? 1 : 0);

        NNMatrix s = df1.negative();

        double d1 = s.negative().dot(s);
        double z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

        while (i < Math.abs(length)) {
            i = i + (length > 0 ? 1 : 0);// count iterations?!

            NNMatrix X0 = inputVector.copy();
            double f0 = f1;
            NNMatrix df0 = df1.copy();
            // CommonOps.addEquals(input.getMatrix(), z1,s.getMatrix());
            // CostGradientTuple evaluateCost2 = f.getCost(inputVector);
            // double f2 = evaluateCost2.getJ();
            // NNMatrix df2 = evaluateCost2.getGrad();
            CommonOps_DDRM.addEquals((DMatrixD1)inputVector.getMatrix(), z1, (DMatrixD1)s.getMatrix());
            NNComputer evaluateCost2 = new NNComputer(nnComputer);
            evaluateCost2.setWeights(vectorToWeights(inputVector));
            evaluateCost2.compute();
            double f2 = evaluateCost2.getCost();
            NNMatrix df2 = NNMatrix.matrixesToVector(evaluateCost2.getGradients());
            LOG.debug(String.format("epoch #%02d: J2 = %f ", i, f2));


            i = i + (length < 0 ? 1 : 0); // count epochs
            double d2 = df2.dot(s);

            double f3 = f1;
            double d3 = d1;
            double z3 = -z1;

            if (length > 0) {
                M = MAX;
            } else {
                M = Math.min(MAX, -length - i);
            }

            int success = 0;
            double limit = -1;

            while(true) {
                while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0)) {
                    // tighten the bracket
                    limit = z1;
                    double z2 = 0.0d;
                    double A = 0.0d;
                    double B = 0.0d;

                    if (f2 > f1) {
                        // quadratic fit
                        z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                    } else {
                        // cubic fit
                        A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                        B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                        // numerical error possible - ok!
                        z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
                    }

                    if (Double.isNaN(z2) || Double.isInfinite(z2)) {
                        // if we had a numerical problem then bisect
                        z2 = z3 / 2.0d;
                    }

                    // don't accept too close to limits
                    z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
                    // update the step
                    z1 = z1 + z2;
                    // CommonOps.addEquals(inputVector.getMatrix(), z2,s.getMatrix());
                    // CostGradientTuple evaluateCost3 = f.getCost(inputVector);
                    // f2 = evaluateCost3.getJ();
                    // df2 = evaluateCost3.getGrad();
                    CommonOps_DDRM.addEquals((DMatrixD1)inputVector.getMatrix(), z2, (DMatrixD1)s.getMatrix());
                    NNComputer evaluateCost3 = new NNComputer(nnComputer);
                    evaluateCost3.setWeights(vectorToWeights(inputVector));
                    evaluateCost3.compute();
                    f2 = evaluateCost3.getCost();
                    df2 = NNMatrix.matrixesToVector(evaluateCost3.getGradients());
                    LOG.debug(String.format("epoch #%02d: J2 = %f ", i, f2));

                    M = M - 1;
                    i = i + (length < 0 ? 1 : 0); // count epochs
                    d2 = df2.dot(s);
                    // z3 is now relative to the location of z2
                    z3 = z3 - z2;
                }

                if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
                    break; // this is a failure
                } else if (d2 > SIG * d1) {
                    success = 1;
                    break; // success
                } else if (M == 0) {
                    break; // failure
                }

                // make cubic extrapolation
                double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));
                // num prob or wrong sign?
                if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0) {
                    // if we have no upper limit
                    if (limit < -0.5) {
                        // the extrapolate the maximum amount
                        z2 = z1 * (EXT - 1);
                    } else {
                        // otherwise bisect
                        z2 = (limit - z1) / 2;
                    }
                } else if ((limit > -0.5) && (z2 + z1 > limit)) {
                    // extraplation beyond max?
                    z2 = (limit - z1) / 2; // bisect
                } else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
                    // extrapolationbeyond limit
                    z2 = z1 * (EXT - 1.0); // set to extrapolation limit
                } else if (z2 < -z3 * INT) {
                    z2 = -z3 * INT;
                } else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
                    // too close to the limit
                    z2 = (limit - z1) * (1.0 - INT);
                }

                // set point 3 equal to point 2
                f3 = f2;
                d3 = d2;
                z3 = -z2;
                z1 = z1 + z2;
                // update current estimates
                // CommonOps.addEquals(inputVector.getMatrix(), z2,s.getMatrix());
                // final CostGradientTuple evaluateCost3 = f.getCost(inputVector);
                // f2 = evaluateCost3.getJ();
                // df2 = evaluateCost3.getGrad();
                CommonOps_DDRM.addEquals((DMatrixD1)inputVector.getMatrix(), z2, (DMatrixD1)s.getMatrix());
                NNComputer evaluateCost3 = new NNComputer(nnComputer);
                evaluateCost3.setWeights(vectorToWeights(inputVector));
                evaluateCost3.compute();
                f2 = evaluateCost3.getCost();
                df2 = NNMatrix.matrixesToVector(evaluateCost3.getGradients());
                LOG.debug(String.format("epoch #%02d: J2 = %f ", i, f2));

                M = M - 1;
                i = i + (length < 0 ? 1 : 0); // count epochs?!
                d2 = df2.dot(s);
            }// end of line search

            NNMatrix tmp = null;
            if (success == 1) { // if line search succeeded
                f1 = f2;

                LOG.debug(String.format("epoch #%02d: J = %f <- success", i, f1));

                // Polack-Ribiere direction: s =
                // (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
                double numerator = (df2.dot(df2) - df1.dot(df2)) / df1.dot(df1);

                s = s.scale(numerator).minus(df2);
                tmp = df1;
                df1 = df2;
                df2 = tmp; // swap derivatives
                d2 = df1.dot(s);
                if (d2 > 0) { // new slope must be negative
                    s = df1.scale(-1.0d); // otherwise use steepest direction
                    d2 = s.scale(-1.0d).dot(s);
                }
                // realmin in octave = 2.2251e-308
                // slope ratio but max RATIO
                z1 = z1 * Math.min(RATIO, d1 / (d2 - 2.2251e-308));
                d1 = d2;
                ls_failed = 0; // this line search did not fail
            } else {
                inputVector = X0;
                f1 = f0;
                df1 = df0; // restore point from before failed line search

                LOG.debug(String.format("epoch #%02d: J = %f <- revert", i, f1));

                // line search failed twice in a row?
                if (ls_failed == 1 || i > Math.abs(length)) {
                    break; // or we ran out of time, so we give up
                }
                tmp = df1;
                df1 = df2;
                df2 = tmp; // swap derivatives
                s = df1.scale(-1.0d); // try steepest
                d1 = s.scale(-1.0d).dot(s);
                z1 = 1.0d / (1.0d - d1);
                ls_failed = 1; // this line search failed
            }

        }

        nnComputer.setWeights(vectorToWeights(inputVector));
    }
}
