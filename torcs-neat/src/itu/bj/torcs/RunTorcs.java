package itu.bj.torcs;

import java.io.IOException;

public class RunTorcs {


	public static void main(String[] args) {

		String[] arguments = {
				"itu.bj.torcs.CNNBot",
				"host:localhost",
				"port:3001",
				"maxEpisodes:1",
				"maxSteps:10000",
				"trackName:aalborg",
				"stage:2"
		};
		Client.setDriver(new CNNBot());
        Client.main(arguments);

		/*for (int i = 0; i < 100; i++){
			TrainingData td = new TrainingData(i, TrainingDataType.MINIMAL);
			try {
				double[] output = Evaluator.getOutput(td.getPixelData());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
*/

	}


	}
