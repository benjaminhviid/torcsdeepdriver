package org.bj.deeplearning.iterator;

import org.bj.deeplearning.dataobjects.TrainingDataHandler;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

/**
 * Created by benjaminhviid on 31/03/2017.
 */
public class CSVIterator extends BaseIterator {

    private static final long serialVersionUID = -1553877073572732933L;
    private int totalExamples = -1;
    private List<String> labels;

    public CSVIterator(int batchSize, int numExamples) {
        super(batchSize, numExamples);
    }

    public CSVIterator(int batchSize, int numExamples, int totalExamples) {
        super(batchSize, numExamples);
        this.totalExamples = totalExamples;
    }


    @Override
    public DataSet next(int num) {
        cursor++;
        return toDataSet(TrainingDataHandler.getRandomTrainingData(num));
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    /**
     * Length of the flattened input array
     */
    @Override
    public int inputColumns() {
        if(inputColumns == -1) {
            inputColumns = TrainingDataHandler.getWidthHeightProduct();
        }
        return inputColumns;
    }

    /**
     * Maximum number of truth values
     */
    @Override
    public int totalOutcomes() {
        if(outputColumns == -1) {
            outputColumns = TrainingData.getFeatureCount();
        }
        return outputColumns;
    }

    @Override
    public List<String> getLabels() {
        if(labels == null) {
            labels = TrainingDataHandler.getGroundTruthLabels();
        }
        return labels;
    }

}