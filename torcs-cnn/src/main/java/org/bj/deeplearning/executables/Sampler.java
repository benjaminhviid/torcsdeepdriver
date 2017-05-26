package org.bj.deeplearning.executables;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.bj.deeplearning.tools.ImageTool;

public class Sampler {

	public static void main(String[] args) throws IOException {
		int imagesToPersist = 250;
		FileSystem.createFolders();
		persistRepresentativeSampleOfDatabase(imagesToPersist);
		System.out.format("Persisted %d sample images in samples folder", imagesToPersist);
	}

	public static void persistRepresentativeSampleOfDatabase(int sampleSize) throws IOException {
		List<TrainingData> randomImages = TrainingDataHandler.getRandomTrainingData(sampleSize);
		
		for(TrainingData td : randomImages) {
			ImageTool.printColoredPngImage(td.getPixelData(), td.getWidth(), new File(FileSystem.getSamplesFolder().resolve("image" + td.getId() + ".png").toString()));
		}
	}
}
