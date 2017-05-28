package org.bj.deeplearning.configuration;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collections;
import java.util.Properties;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.bj.deeplearning.tools.PropertiesReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ContinuousSequentialTraining extends ContinuousTraining {
    //final JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local[*]").set("spark.driver.maxResultSize","3g")
      //      .setAppName("scenes"));


    public ContinuousSequentialTraining() throws IOException {
        super();
    }

    @Override
    public void train(DataSetIterator trainIterator, DataSetIterator testIterator) {


        if (PropertiesReader.getProjectProperties().getProperty("training.collectStats").equals("true")) {
            for (int i = latestEpoch; i <= nEpochs; i++) {


                if (visualize) {
                    //Initialize the user interface backend
                    UIServer uiServer = UIServer.getInstance();

                    //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
                    //Then add the StatsListener to collect this information from the network, as it trains
                    StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
                    int listenerFrequency = 1;
                    model.setListeners(new StatsListener(statsStorage, listenerFrequency));

                    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
                    uiServer.attach(statsStorage);
                }
                model.fit(trainIterator);
                System.out.println(String.format("*** Completed epoch %d ***", i));
                testIterator.reset();

                saveModel(model, i);
                outputDeadNeurons(model);
            }
        }
    }
}