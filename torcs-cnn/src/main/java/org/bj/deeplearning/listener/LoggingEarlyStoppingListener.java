package org.bj.deeplearning.listener;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author: Ousmane A. Dia
 */
public class LoggingEarlyStoppingListener implements EarlyStoppingListener<MultiLayerNetwork> {

    private static Logger log = LoggerFactory.getLogger(LoggingEarlyStoppingListener.class);
    private int onStartCallCount = 0;
    private int onEpochCallCount = 0;
    private int onCompletionCallCount = 0;

    @Override
    public void onStart(EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration, MultiLayerNetwork multiLayerNetwork) {
        log.info("EarlyStopping: onStart called");
        onStartCallCount++;
    }

    @Override
    public void onEpoch(int i, double v, EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration, MultiLayerNetwork multiLayerNetwork) {
        log.info("EarlyStopping: onEpoch called (epochNum={}, score={}}", i, v);
        onEpochCallCount++;
    }

    @Override
    public void onCompletion(EarlyStoppingResult<MultiLayerNetwork> esResult) {
        log.info("EarlyStopping: onCompletion called (result: {})", esResult);
        onCompletionCallCount++;
    }

    public int getOnCompletionCallCount() {
        return onCompletionCallCount;
    }

    public int getOnStartCallCount() {
        return onStartCallCount;
    }

    public int getOnEpochCallCount() {
        return onEpochCallCount;
    }

}