package org.bj.deeplearning.configuration;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface Trainable {
	void train(DataSetIterator trainIterator, DataSetIterator testIterator);
}
