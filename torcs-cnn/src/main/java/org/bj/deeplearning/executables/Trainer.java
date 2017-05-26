package org.bj.deeplearning.executables;

import java.io.IOException;

import org.bj.deeplearning.configuration.ContinuousSequentialTraining;
import org.bj.deeplearning.configuration.EarlyStoppingTraining;
import org.bj.deeplearning.configuration.Trainable;
import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.RunType;
import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.bj.deeplearning.iterator.CSVIterator;
import org.bj.deeplearning.tools.PropertiesReader;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class Trainer {

	private static int testSize, validationSize, trainSize, batchSize;
	private static boolean checkIntegrity;
	private static String trainingPersistenceType, trainingType;

	public static void main(String[] args) throws IOException {
		init();
		doTrain();
	}

	private static void configureND4J() {
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
	}

	private static void doTrain() throws IOException {
		TrainingDataHandler.runType = RunType.TRAINING;
		DataSetIterator trainIterator = getIterator(batchSize, trainSize);
		DataSetIterator testIterator = getIterator(batchSize, testSize);

		getTrainer().train(trainIterator, testIterator);
	}

	private static DataSetIterator getIterator(int batchSize, int numExamples) {
		return new CSVIterator(batchSize, numExamples);

	}

	private static Trainable getTrainer() throws IOException {
		switch (trainingType) {
			case "sequential":
				return new ContinuousSequentialTraining();
			case "early-stopping":
				return new EarlyStoppingTraining();
			default:
				return null;
		}
	}

	public static void init() throws IOException {
		configureND4J();

		testSize = intFromProperty("training.testSize");
		validationSize = intFromProperty("training.validationSize");
		trainSize = intFromProperty("training.trainSize");
		batchSize = intFromProperty("training.persistence.batchSize");

		checkIntegrity = boolFromProperty("training.persistence.checkIntegrity");

		trainingPersistenceType = stringFromProperty("training.persistence.type");
		trainingType = stringFromProperty("training.type");

		checkProperties();

		FileSystem.createFolders();
		if(trainingPersistenceType.equals("filesystem")) {
			FileSystem.persist(trainSize, validationSize, testSize, batchSize, checkIntegrity);
		}
	}

	private static void checkProperties() {
		boolean exit = false;

		if(trainingPersistenceType == null) {
			System.out.println("Please provide an option for \"training.persistence.type\" in the properties.");
			System.out.println("Currently \"filesystem\" and \"database\" is supported");
			exit = true;
		}
		if(trainingType == null) {
			System.out.println("Please provide an option for \"training.type\" in the properties.");
			System.out.println("Currently \"parallel\", \"sequential\" and \"early-stopping\" is supported");
			exit = true;
		}

		if(exit) {
			System.exit(12321);
		}
	}

	private static int intFromProperty(String property) {
		return Integer.parseInt(PropertiesReader.getProjectProperties().getProperty(property));
	}

	private static boolean boolFromProperty(String property) {
		return "true".equals(PropertiesReader.getProjectProperties().getProperty(property));
	}

	private static String stringFromProperty(String property) {
		return PropertiesReader.getProjectProperties().getProperty(property);
	}

	public static int lastTrainIndex() {
		return trainSize;
	}

	public static int lastValidationIndex() {
		return lastTrainIndex()+validationSize;
	}

	public static int lastTestIndex() {
		//return lastValidationIndex() + testSize;
		return testSize;


	}
}
