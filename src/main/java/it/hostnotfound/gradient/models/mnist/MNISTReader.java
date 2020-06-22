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

package it.hostnotfound.gradient.models.mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.File;

import java.nio.file.Files;
import java.nio.file.Paths;

import java.util.zip.GZIPInputStream;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reader for the MNIST files
 */
public class MNISTReader {
    private static final Logger LOG = LoggerFactory.getLogger(MNISTReader.class);

    protected static String MNIST_DIR = "mnist";

    protected static final String TRAIN_IMAGES_FILE = "train-images-idx3-ubyte.gz";
    protected static final String TRAIN_LABELS_FILE = "train-labels-idx1-ubyte.gz";
    protected static final String TEST_IMAGES_FILE = "t10k-images-idx3-ubyte.gz";
    protected static final String TEST_LABELS_FILE = "t10k-labels-idx1-ubyte.gz";

    /**
     * Constructor which uses the default MNIST directory path
     * 
     * @throws FileNotFoundException if the directory is not found
     */
    public MNISTReader() throws FileNotFoundException {
        initialize();
    }

    /**
     * Constructor
     * 
     * @param mnistDirectory the path of MNIST directory containing the data set files
     * @throws FileNotFoundException if the directory is not found
     */
    public MNISTReader(String mnistDirectory) throws FileNotFoundException {
        MNIST_DIR=mnistDirectory;
        initialize();
    }

    private void initialize() throws FileNotFoundException {
        if (!Files.isDirectory(Paths.get(MNIST_DIR))) {
            throw new FileNotFoundException("The directory '" + MNIST_DIR + "' cannot be found in the current path");
        }
    }

    /**
     * Read the images and labels from the training set files
     * 
     * @return list of MNISTItem objects containing both image and label
     * @throws IOException
     */
    public ArrayList<MNISTItem> readTrainSet() throws IOException {
        String trainImagesFile = MNIST_DIR + (String) File.separator + TRAIN_IMAGES_FILE;
        String trainLabelsFile = MNIST_DIR + (String) File.separator + TRAIN_LABELS_FILE;

        if (! Files.exists(Paths.get(trainImagesFile))) {
            throw new FileNotFoundException("File '" + trainImagesFile + "' not found");
        }

        if (! Files.exists(Paths.get(trainLabelsFile))) {
            throw new FileNotFoundException("File '" + trainLabelsFile + "' not found");
        }

        ArrayList<MNISTItem> trainSet = MNISTReader.readData(trainImagesFile, trainLabelsFile);

        LOG.info("Successfully read " + trainSet.size() + " labelled images from the training set");

        return trainSet; 
    }

    /**
     * Read the images and labels from the test set files
     * 
     * @return list of MNISTItem objects containing both image and label
     * @throws IOException
     */
    public ArrayList<MNISTItem> readTestSet() throws IOException {
        String testImagesFile = MNIST_DIR + (String) File.separator + TEST_IMAGES_FILE;
        String testLabelsFile = MNIST_DIR + (String) File.separator + TEST_LABELS_FILE;

        if (! Files.exists(Paths.get(testImagesFile))) {
            throw new FileNotFoundException("File '" + testImagesFile + "' not found");
        }

        if (! Files.exists(Paths.get(testLabelsFile))) {
            throw new FileNotFoundException("File '" + testLabelsFile + "' not found");
        }

        ArrayList<MNISTItem> testSet = MNISTReader.readData(testImagesFile, testLabelsFile);

        LOG.info("Successfully read " + testSet.size() + " labelled images from the test set");

        return testSet;
    }

    /**
     * Read data from MNIST files
     * 
     * @param imagesFile path of the gzipped file containing the images
     * @param labelsFile path of the gzipped file containing the labels
     * @return list of MNISTItem objects, each of them representing image
     *         information and their label
     * @throws IOException
     */
    public static ArrayList<MNISTItem> readData(String imagesFile, String labelsFile) throws IOException {
        DataInputStream imagesInputStream = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(imagesFile))));
        // ---- images file format ----
        //
        // [offset] [type]          [value]          [description]
        // 0000     32 bit integer  0x00000803(2051) magic number
        // 0004     32 bit integer  ??               number of images
        // 0008     32 bit integer  28               number of rows
        // 0012     32 bit integer  28               number of columns
        // 0016     unsigned byte   ??               pixel
        // 0017     unsigned byte   ??               pixel
        // ........
        // xxxx     unsigned byte   ??               pixel

        int imagesMagicNumber = imagesInputStream.readInt();
        int numberOfImages = imagesInputStream.readInt();
        int numberOfRows = imagesInputStream.readInt();
        int numberOfColumns = imagesInputStream.readInt();

        LOG.debug("images file - magic number: " + imagesMagicNumber);
        LOG.debug("images file - number of images: " + numberOfImages);
        LOG.debug("images file - number of rows: " + numberOfRows);
        LOG.debug("images file - number of columns: " + numberOfColumns);

        DataInputStream labelsInputStream = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(labelsFile))));
        // ---- labels file format ----
        //
        // [offset] [type]          [value]          [description]
        // 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        // 0004     32 bit integer  ??               number of items
        // 0008     unsigned byte   ??               label
        // 0009     unsigned byte   ??               label
        // ........
        // xxxx     unsigned byte   ??               label

        int labelsMagicNumber = labelsInputStream.readInt();
        int numberOfLabels = labelsInputStream.readInt();

        LOG.debug("labels file - magic number: " + labelsMagicNumber);
        LOG.debug("labels file - number of labels: " + numberOfLabels);
        
        assert numberOfImages == numberOfLabels;

        ArrayList<MNISTItem> data = new ArrayList<MNISTItem>(numberOfImages);
        int numberOfPixels = numberOfRows*numberOfColumns;

        for(int i = 0; i < numberOfImages; i++) {
            byte[] pixels = new byte[numberOfPixels];
            for (int j = 0; j < numberOfPixels; j++) {
                pixels[j] = imagesInputStream.readByte();
            }

            MNISTItem mnistItem = new MNISTItem();
            mnistItem.setWidth(numberOfColumns);
            mnistItem.setHeight(numberOfRows);
            mnistItem.setLabel(labelsInputStream.readByte());
            mnistItem.setPixels(pixels);

            data.add(mnistItem);
        }

        imagesInputStream.close();
        labelsInputStream.close();

        return data;
    }
}
