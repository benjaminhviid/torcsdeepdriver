package itu.bj.torcs;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;


import org.jgap.BulkFitnessFunction;
import org.jgap.Chromosome;

import com.anji.integration.ActivatorTranscriber;
import com.anji.util.Configurable;
import com.anji.util.Properties;

public class TorcsFitnessFunction implements BulkFitnessFunction, Configurable {

	private final static String TRANSCRIBER_CLASS_KEY = "torcsai.transcriber";
	private ActivatorTranscriber activatorFactory = new ActivatorTranscriber();
	public boolean recordSession = false;
	private int dataCollectionInterval = 100; // in ms

	private TorcsNeatAI bestBot;
	private PrintWriter writer;

	public void init(Properties props) throws Exception {
		// TODO Auto-generated method stub
		activatorFactory = (ActivatorTranscriber) props.newObjectProperty( TRANSCRIBER_CLASS_KEY );
		bestBot = null;
		
	}
	
	public void evaluate(List subjects) {
		// TODO Auto-generated method stub
		System.out.println("evaluating");
		TorcsNeatAI[] bots = new TorcsNeatAI[subjects.size()];
		double bestScore = 0.0;
		try{
			writer = new PrintWriter("generation.txt", "UTF-8");
			if (recordSession)
				DataCollector.StartDataCollection(dataCollectionInterval);

			for (int i = 0; i < subjects.size(); i++){
				try{
					TorcsNeatAI bot = new TorcsNeatAI(activatorFactory.newActivator((Chromosome)subjects.get(i)));
					bot.runGame();
					bots[i] = bot;
					writer.println(i + ": score " + bot.fitness);
					((Chromosome)subjects.get(i)).setFitnessValue(bot.fitness.intValue());
				} catch(Exception e){
					System.out.println("Attention: " + subjects.get(i).toString() + "\n" + e.getMessage());
					System.exit(0);
				}
			}
			
			Arrays.sort(bots);
			bestBot = bots[bots.length-1];
			writer.println("\nBest score: "+bestBot.fitness);
			writer.close();
			
		} catch (Exception e){
			System.out.println(e.getMessage());
		}
		
	}

	public int getMaxFitnessValue() {
		// TODO Auto-generated method stub
		return 2580; // aalborg meters
	}
}