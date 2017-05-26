package org.bj.deeplearning.iterator;

import java.util.List;

import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.nd4j.linalg.dataset.DataSet;

public class FileSystemIterator extends BaseIterator {

	private static final long serialVersionUID = 4292186710513240524L;

	public FileSystemIterator(int batchSize, int numExamples) {
		super(batchSize, numExamples);
	}

	@Override
	public DataSet next(int num) {
		cursor++;
		return toDataSet(FileSystem.getRandomImages(num));
	}

	@Override
	public int totalExamples() {
		return TrainingDataHandler.getTotalNumberOfImages();
	}

	/**
	 * Length of the flattened input array
	 */
	@Override
	public int inputColumns() {
		if(inputColumns == -1) {
			TrainingData td = FileSystem.load(1);
			inputColumns = td.getWidth() * td.getHeight() * 3;
		}
		return inputColumns;
	}

	/**
	 * Maximum number of truth values
	 */
	@Override
	public int totalOutcomes() {
		if(outputColumns == -1) {
			outputColumns = FileSystem.load(1).getFeatures().length;
		}
		return outputColumns;
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException();
	}
}
