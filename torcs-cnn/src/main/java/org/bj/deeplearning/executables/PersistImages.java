package org.bj.deeplearning.executables;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.IntStream;

import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.bj.deeplearning.tools.ImageTool;

public class PersistImages {

	private static final String imagesFolder = "images";
	private static final String imagesWithBotsFolder = "images-with-bots";
	private static final String imagesWithoutBotsFolder = "images-without-bots";
	private static final int batchDownloadSize = 250;

	public static void main(String[] args) throws IOException {
		int samples = 100000;
		int fromId = 1;
		int toId = fromId + samples - 1;

		FileSystem.createFolders();
		for(int id = fromId; id <= toId; id += batchDownloadSize) {
			int endId = Math.min(toId, id + batchDownloadSize - 1);
			List<TrainingData> trainingData = TrainingDataHandler.getTrainingData(id, endId);
			FileSystem.persist(trainingData);
		}

		createFolders();
		saveImages(fromId, toId);
	}

	public static void saveImages(int fromId, int toId) {
		System.out.println(String.format("Persisting % 5d images", toId - fromId + 1));

		IntStream.rangeClosed(fromId, toId).boxed().forEach(id -> saveImage(id));
	}

	public static void saveImages(int upToId) throws IOException {
		System.out.println(String.format("Persisting % 5d images", upToId));

		IntStream.rangeClosed(1, upToId).boxed().forEach(id -> saveImage(id));
	}

	public static void saveImage(int id) {
		try {
		TrainingData td = FileSystem.load(id);
		//if(td.isWithinSight()) {
		//	ImageTool.printColoredPngImage(td.getPixelData(), td.getWidth(), new File(imagesWithBotsFolder + File.separator + td.getId() + ".png"));
		//} else {
			ImageTool.printColoredPngImage(td.getPixelData(), td.getWidth(), new File(imagesWithoutBotsFolder + File.separator + td.getId() + ".png"));
		//}
		ImageTool.printColoredPngImage(td.getPixelData(), td.getWidth(), new File(imagesFolder + File.separator + td.getId() + ".png"));
		} catch(IOException e) {
			e.printStackTrace();
			System.exit(42);
		}
	}

	public static void createFolders() {
		new File(imagesFolder).mkdirs();
		new File(imagesWithBotsFolder).mkdirs();
		new File(imagesWithoutBotsFolder).mkdirs();
	}
}
