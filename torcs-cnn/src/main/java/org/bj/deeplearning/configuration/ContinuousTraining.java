package org.bj.deeplearning.configuration;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.Time;
import java.time.Instant;
import java.util.Properties;

import org.bj.deeplearning.listener.ScoreLogListener;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.dataobjects.TrainingData;
import org.bj.deeplearning.executables.DeadNeuronDetector;
import org.bj.deeplearning.listener.IterationTimeListener;
import org.bj.deeplearning.tools.PropertiesReader;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public abstract class ContinuousTraining implements Trainable {

	protected MultiLayerNetwork model;
	protected MultiLayerConfiguration configuration;
	protected int height, width, featureCount, nEpochs, latestEpoch = 0;
	private boolean outputDeadNeurons, saveModel, collectStats, disableStatsWhenTraining;
    File statsFile;

	public ContinuousTraining() throws IOException {
		init();
	}

	@Override
	public void train(DataSetIterator trainIterator, DataSetIterator testIterator) {

        if (collectStats) {
            for (int i = latestEpoch; i <= nEpochs; i++) {
                model.fit(trainIterator);
                System.out.println(String.format("*** Completed epoch %d ***", i));
                testIterator.reset();
                saveModel(model, i);
                outputDeadNeurons(model);
            }
        }
        else{
/*
            //Second run: Load the saved stats and visualize. Go to http://localhost:9000/train
            StatsStorage statsStorage = new FileStatsStorage(statsFile);    //If file already exists: load the data from it
            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);*/
        }
	}

	protected void saveModel(MultiLayerNetwork model, int numeration) {
        if(saveModel) {
            try {
				ModelSerializer.writeModel(model, new File(FileSystem.getPathOfModelFile(numeration).toString()), true);
            } catch (IOException e) {
				e.printStackTrace();
			}
        }
	}

	protected void outputDeadNeurons(MultiLayerNetwork model) {
		if(outputDeadNeurons) {
			DeadNeuronDetector.getDeadNeurons(model, 100);
		}
	}

	protected void init() throws IOException {
		Properties projectProperties = PropertiesReader.getProjectProperties();
		width = Integer.parseInt(projectProperties.getProperty("training.image.width"));
		height = Integer.parseInt(projectProperties.getProperty("training.image.height"));
		featureCount = TrainingData.getFeatureCount();
        nEpochs = Integer.parseInt(projectProperties.getProperty("training.epochs"));
        outputDeadNeurons = projectProperties.getProperty("training.outputDeadNeurons").equals("true");
        saveModel = projectProperties.getProperty("training.saveModel").equals("true");
        collectStats = projectProperties.getProperty("training.collectStats").equals("true");
        disableStatsWhenTraining = projectProperties.getProperty("training.collectStats").equals("true");


        initConfig();
        initNetwork();


	}

	protected void initConfig() {
		configuration = BuilderFactory.getDeepConvNet(height, width, featureCount).build();

	}

	protected void initNetwork() throws IOException {
		if(PropertiesReader.getProjectProperties().getProperty("training.continueFromLatestModel").equals("true")) {
			model = ModelSerializer.restoreMultiLayerNetwork(new FileInputStream(FileSystem.getPathOfLatestModelFile().toString()));
			latestEpoch = FileSystem.findLatestModelId();
		} else {
			model = new MultiLayerNetwork(configuration);
	        model.init();
		}

        statsFile = new File(FileSystem.getContinuousFolder() + "/UIStorageExampleStats.dl4j");

		if (collectStats) {
            StatsStorage statsStorage = new FileStatsStorage(statsFile);
            if (disableStatsWhenTraining)
               model.setListeners(new IterationTimeListener(), new ScoreIterationListener(), new ScoreLogListener(100, "deepnet"+ Instant.now().toString()));
				//model.setListeners(new IterationTimeListener(), new ScoreIterationListener());
			else
                model.setListeners(new IterationTimeListener(), new StatsListener(statsStorage), new ScoreIterationListener());
        }

    }
}
