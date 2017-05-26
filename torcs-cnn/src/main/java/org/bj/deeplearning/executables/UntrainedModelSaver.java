package org.bj.deeplearning.executables;

import java.io.IOException;
import java.util.Properties;

import org.bj.deeplearning.configuration.BuilderFactory;
import org.bj.deeplearning.dataobjects.FileSystem;
import org.bj.deeplearning.tools.PropertiesReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.bj.deeplearning.dataobjects.TrainingData;

public class UntrainedModelSaver {
	public static void main(String[] args) {
		Properties projectProperties = PropertiesReader.getProjectProperties();
		int featureCount = TrainingData.getFeatureCount();
		int width = Integer.parseInt(projectProperties.getProperty("training.image.width"));
		int height = Integer.parseInt(projectProperties.getProperty("training.image.height"));

		MultiLayerConfiguration configuration = BuilderFactory.getVeryShallowConvNet(height, width, featureCount).build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

		try {
			ModelSerializer.writeModel(model, FileSystem.getModelsFolder().resolve("untrainedModel.bin").toFile(), true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println("Saved untrained model");
	}
}
