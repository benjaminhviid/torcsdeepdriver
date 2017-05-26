package org.bj.deeplearning.executables;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import com.sun.org.apache.xpath.internal.operations.Mult;
import org.apache.commons.lang.ArrayUtils;
import org.bj.deeplearning.dataobjects.RunType;
import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.bj.deeplearning.listener.ScoreLogListener;
import org.bj.deeplearning.tools.INDArrayTool;
import org.bj.deeplearning.tools.NNTool;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.bj.deeplearning.tools.ImageTool;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class Evaluator {

	private static final int BATCH_SIZE = 32;
	private static MultiLayerNetwork net;
	public static void main(String[] args) throws IOException {
		Trainer.init();
		//FileSystem.createModelsFolders();
		Path modelPath;
		if(args.length != 1) {
			modelPath = FileSystem.getPathOfLatestModelFile();
		} else {
			modelPath = FileSystem.getPathOfModelFile(Integer.parseInt(args[0]));
		}
		TrainingDataHandler.runType = RunType.TRAINING;

		MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(new FileInputStream(modelPath.toFile()));
		network.setListeners(new ScoreLogListener(10, "evaluator"));
		//System.out.println(network.getLayerWiseConfigurations().toString());
		//System.out.format("Evaluating model %s \n", modelPath.toString());
		double[] accuracy = getAccuracy(network);

		for (double d : accuracy){
		    System.out.print (d + " ");
        }
	}




	public static double[] getAccuracy(MultiLayerNetwork network) throws IOException {
		Files.createDirectories(Paths.get("misses"));
		//int firstId = Trainer.lastValidationIndex()+1;
		int firstId = 1;
		int lastId = Trainer.lastTestIndex();
		if(lastId-firstId < 0) {
			throw new IllegalArgumentException("Test set is empty");
		}

		@SuppressWarnings("unused")
		int hits = 0, misses = 0;
        double[] accuracy = new double[]{0.0,0.0};

		List<TrainingData> testSet;
		for(int fromId = firstId; fromId <= lastId; fromId += BATCH_SIZE) {
			int toId = Math.min(fromId + BATCH_SIZE, lastId);
			testSet = FileSystem.load(fromId, toId);
			for(TrainingData td : testSet) {
			    System.out.println("Test id: " + td.getId());
				double[] groundTruths = td.getFeatures();
                for (double d : groundTruths){
                    System.out.print (String.format( "%.2f", d ) + " ");
                }
                System.out.println("");
                INDArray in = Nd4j.create(ImageTool.toScaledDoubles(td.getPixelData()), new int[] {1, 3, td.getWidth(), td.getHeight()});
                INDArray output = network.output(in, false);

                for (double d : INDArrayTool.toFlatDoubleArray(output)){
                    System.out.print (String.format( "%.2f", d ) + " ");
                }
                System.out.println("");
                System.out.println("");

                for (int i = 0; i < 2; i++){ // TODO do such that it allows other than output length 4
                	accuracy[i] =  Math.abs(groundTruths[i] - INDArrayTool.toFlatDoubleArray(output)[i]);
                }

                // System.out.println(output);

			}
			System.out.println(String.format("Evaluated %d images", toId-firstId));
		}

        for (double d: accuracy) {
            d = d / ((double) lastId-firstId+1);
        }
        return accuracy;

	}

	public static double[] getOutput(byte[] pixels) throws IOException {

		if (net == null){
			setNetwork();
		}

		INDArray in = Nd4j.create(ImageTool.toScaledDoubles(pixels), new int[] {1, 3, TrainingDataHandler.getWidth(), TrainingDataHandler.getHeight()});
		INDArray out = net.output(in, false);

		double[] output = new double[out.length()];

		for(int i = 0; i < out.length(); i++) {
			output[i] = out.getDouble(i);
			//System.out.print(output[i] + " ");
		}
		return output;
	}

	private static void setNetwork() throws IOException{
		Path modelPath = FileSystem.getPathOfLatestModelFile();
		net = ModelSerializer.restoreMultiLayerNetwork(new FileInputStream(modelPath.toFile()));
	}

	private static int getIndex(double[] arr) {
		for(int idx = 0; idx < arr.length; idx++) {
			if(1d - arr[idx] < 1E-6) {
				return idx;
			}
		}
		throw new RuntimeException("Did not found a 1 in the array");
	}

	public static boolean almostEqual(double a, double b, double eps){
		return Math.abs(a-b)<eps;
	}
}
