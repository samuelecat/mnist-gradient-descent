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

package it.hostnotfound.gradient.models.minimizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import it.hostnotfound.gradient.models.NNComputer;
import it.hostnotfound.gradient.models.NNMatrix;

public class Naive implements Minimizer {
    private final static Logger LOG = LoggerFactory.getLogger(Naive.class);

    private NNComputer nnComputer;
    private int maxEpoch;
    private double alpha;

    private double costs[];
    private NNComputer previous;
    private final double epsilon = 1.0/10000L;

    public Naive(final NNComputer nnComputer, final int maxEpoch, final double alpha) {
        this.nnComputer = nnComputer;
        this.maxEpoch = maxEpoch;
        this.alpha = alpha;
    }

    @Override
    public void minimize() {
        costs = new double[maxEpoch];
        previous = new NNComputer(nnComputer);

        for(int epoch=0; epoch < maxEpoch; epoch++) {
            nnComputer.compute();
            costs[epoch] = nnComputer.getCost();
            LOG.debug(String.format("epoch #%02d: J = %.10f", epoch, costs[epoch]));

            if ((epoch > 0) && ((costs[epoch] + epsilon) >= costs[epoch - 1])) {
                LOG.debug("minimum cost reached");
                nnComputer.setWeights(previous.getWeights());
                break;
            }

            final NNMatrix[] gradients = nnComputer.getGradients();
            final NNMatrix[] Weights = nnComputer.getWeights();
            // Wj = Wj − a*∂J(W)/∂Wj
            Weights[0] = Weights[0].plus(gradients[0].scale(-alpha));
            Weights[1] = Weights[1].plus(gradients[1].scale(-alpha));

            previous = new NNComputer(nnComputer);
            nnComputer.setWeights(Weights);
        }
    }

}
