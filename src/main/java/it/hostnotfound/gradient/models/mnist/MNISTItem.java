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

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.awt.image.DataBufferByte;
import java.awt.RenderingHints;

/**
 * Holds both the image data and its label
 */
public class MNISTItem implements java.io.Serializable {

    private static final long serialVersionUID = 1L;
    private int width;
    private int height;
    private int label;
    private byte[] pixels;

    public MNISTItem() {
    }

    /**
     * Get the image
     * 
     * @return the image as BufferImage object
     */
    public BufferedImage getImage() {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = (WritableRaster) image.getData();
        int[] intPixels = getIntPixels();
        // for (int i=0; i < intPixels.length; i++) {
        //     System.out.printf("%1$3s,", intPixels[i]);
        //     if (i > 0 && ((i+1) % width == 0)) {
        //         System.out.println("");
        //     }
        // }
        raster.setPixels(0, 0, width, height, intPixels);
        image.setData(raster);
        return image;
    }

    /**
     * Scale image
     * 
     * @param scaleWidth desired width
     * @param scaleHeight desired height
     */
    public void scale(double scaleWidth, double scaleHeight) {
        int newWidth = (int) (scaleWidth * width);
        int newHeight = (int) (scaleHeight * height);
        BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(getImage(), 0, 0, newWidth, newHeight, 0, 0, width, height, null);
        g.dispose();
        // store the pixels back
        pixels = ((DataBufferByte) resized.getRaster().getDataBuffer()).getData();
        width = newWidth;
        height = newHeight;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height=height;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label=label;
    }

    /**
     * Get pixels as *unsigned* bytes
     * 
     * @return pixels as unsigned values 0-255
     */
    public byte[] getPixels() {
        return pixels.clone();
    }

    /**
     * Get pixels as int
     * 
     * @return pixels as int
     */
    public int[] getIntPixels() {
        int[] intPixels = new int[pixels.length];
        for (int i=0; i < pixels.length; i++) {
            // unsigned byte -> int
            intPixels[i] = ((int) pixels[i]) & 0xFF;
        }
        return intPixels;
    }

    /**
     * Get pixels as double
     * 
     * @return pixels as double
     */
    public double[] getDoublePixels() {
        double[] intPixels = new double[pixels.length];
        for (int i=0; i < pixels.length; i++) {
            // unsigned byte -> int -> double
            intPixels[i] = (double) (((int) pixels[i]) & 0xFF);
        }
        return intPixels;
    }

    /**
     * Set pixels as unsigned bytes
     * 
     * @param pixels array of unsigned bytes (value 0-255)
     */
    public void setPixels(byte[] pixels) {
        this.pixels = new byte[pixels.length];
        System.arraycopy(pixels, 0, this.pixels, 0, pixels.length);
    }

    /**
     * Get the total count of pixel values
     * 
     * @return total pixels
     */
    public int size() {
        return pixels.length;
    }

}
