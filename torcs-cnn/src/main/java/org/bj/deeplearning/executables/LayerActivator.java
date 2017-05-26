package org.bj.deeplearning.executables;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.bj.deeplearning.tools.ImageTool;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.bj.deeplearning.tools.INDArrayTool;
import org.bj.deeplearning.tools.NNTool;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.bj.deeplearning.dataobjects.TrainingDataHandler.imgLoader;

public class LayerActivator {
	public static void main(String[] args) throws IOException {
		FileSystem.createFolders();
		/*if(args.length != 2) {
			System.err.println("Expected 2 arguments, the model number, and the id of the image to evaluate on.");
			System.exit(1);
		}
		*/
		File networkFile = Paths.get("models", "continuous", "model0.bin").toFile();
		int[] imageIds = new int[] { 45 };
		
		for(int imageId : imageIds) {
			getActivationOfLayers(networkFile, imageId);
		}
	}																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										
	
	public static void getActivationOfLayers(File networkFile, int imageId) throws IOException {
		MultiLayerNetwork network =	 ModelSerializer.restoreMultiLayerNetwork(new FileInputStream(networkFile));

		INDArray indArray = imgLoader.asMatrix(new File(TrainingDataHandler.SCREENSHOTS_PATH + "screenshot" + imageId + ".jpg"));
		TrainingData image = TrainingDataHandler.getTrainingData(imageId, imageId).get(0);
	//	ImageTool.printColoredPngImage(image.getPixelData(), image.getWidth(), new File("image"+ imageId +".png"));
	//	System.out.println("Image ID: " + imageId + " pixeldata length " + image.getPixelData().length);
		double[] pd = INDArrayTool.toFlatDoubleArray(indArray);
		double[] pixeldata = ImageTool.normalize(pd);

		network.output(Nd4j.create(pixeldata, new int[] { 1, 3, TrainingDataHandler.getWidth(), TrainingDataHandler.getHeight()}), false);
		int convLayers = NNTool.numberOfConvolutionalLayers(network);


		//Save convolutional activations as images
		for(int currentLayer = 0; currentLayer < convLayers; currentLayer++) {
			persistFeatureMap(network.getLayer(currentLayer).activate().slice(0), currentLayer);
		}
		
		System.out.println();
		System.out.println("Evaluating image with id: " + imageId);
		
		//Print dense layer and output layer activation
		double[] output = null;
		for(int currentLayer = convLayers; currentLayer < network.getLayers().length; currentLayer++) {
			Layer layer = network.getLayer(currentLayer);
			INDArray activate = layer.activate();
			if(currentLayer+1 == network.getLayers().length) {
				output = INDArrayTool.toFlatDoubleArray(activate);
			}
		}
		
		System.out.println("Output of output layer:");
		System.out.println(Arrays.toString(output));
		System.out.println("Output of ground truths:");
		System.out.println(Arrays.toString(image.getFeatures()));
		
		Pair<Integer, Double> featureInfo = findHighestIdAndValue(image.getFeatures());
		System.out.println("Correct index = " + featureInfo.getLeft());
		
		Pair<Integer, Double> networkInfo = findHighestIdAndValue(output);
		System.out.println("Guessed index = " + networkInfo.getLeft());
		System.out.println("Guessed percentage = " + networkInfo.getRight());
		System.out.println("Percentage of correct index = " + output[featureInfo.getLeft()]);
		
		System.out.println();
	}
	
	private static void persistFeatureMap(INDArray featureMaps, int layer) throws IOException {

		for(int featureMap = 0; featureMap < featureMaps.shape()[0]; featureMap++) {
			ImageTool.printGreyScalePngImage(INDArrayTool.toFlatDoubleArray(featureMaps.slice(featureMap)), featureMaps.shape()[1], FileSystem.getFeatureMapsFolderForLayer(layer).resolve("featureMap" + featureMap + ".png").toFile());
/*
			byte[] arr = new byte[14400];
			int counter = 0;
			for (Double d: INDArrayTool.toFlatDoubleArray(featureMaps.slice(featureMap))){
				arr[counter] = ImageTool.toByte(d);
				counter++;
			}

			ImageTool.printColoredPngImage(arr,featureMaps.shape()[1], FileSystem.getFeatureMapsFolderForLayer(layer).resolve("featureMap" + featureMap + ".png").toFile());

		*/}
	}
	
	private static Pair<Integer, Double> findHighestIdAndValue(double[] arr) {
		int idx = 0;
		double val = 0d;
		for(int i = 0; i < arr.length; i++) {
			if(val < arr[i]) {
				val = arr[i];
				idx = i;
			}
		}
		return new ImmutablePair<Integer, Double>(idx, val);
	}
}










