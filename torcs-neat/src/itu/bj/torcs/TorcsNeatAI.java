package itu.bj.torcs;


import java.io.IOException;
import java.util.List;

import org.jgap.Chromosome;
import org.jgap.Genotype;
import org.jgap.InvalidConfigurationException;

import com.anji.integration.Activator;
import com.anji.integration.ActivatorTranscriber;
import com.anji.integration.TranscriberException;
import com.anji.neat.NeatConfiguration;
import com.anji.persistence.FilePersistence;
import com.anji.persistence.Persistence;
import com.anji.util.Properties;

public class TorcsNeatAI implements Comparable<TorcsNeatAI>{

	public Double fitness;
	private Activator a;
	private Properties properties;
	private final static String TRANSCRIBER_CLASS_KEY = "torcsai.transcriber";
	private final static String CHROMOSOME_ID_KEY = "torcsai.chromosome.id";
	
	public TorcsNeatAI(Activator a){
		this.a = a;
		fitness = 0.0;
	}
	
	
	public TorcsNeatAI() {
		try{
			properties = new Properties("torcsai.properties");
			NeatConfiguration configuration = new NeatConfiguration(properties);

			Persistence persistence = new FilePersistence();
			persistence.init(properties);
			String chromosome_id = properties.getProperty(CHROMOSOME_ID_KEY);
			Chromosome chromosome = persistence.loadChromosome(chromosome_id, configuration);

			ActivatorTranscriber activator_transcriber = (ActivatorTranscriber)properties.newObjectProperty(TRANSCRIBER_CLASS_KEY);
			a = activator_transcriber.newActivator(chromosome);
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}

	 
	  
	public void runGame() {
		// TODO Auto-generated method stub
	
	
		System.out.println("Running simulation...");
		
		String[] arguments = {
				"itu.bj.torcs.TorcsNeatController",
				"host:localhost", 
				"port:3001", 
				"maxEpisodes:1", 
				"maxSteps:10000",
				"trackName:aalborg", 
				"stage:2"
				};
		Client.setDriver(new TorcsNeatController(a));
		Client.main(arguments);

		fitness = Client.getFinalScore();
		System.out.print("Score: ");
		System.out.printf("%.4f", fitness);
		System.out.println("");
		
		
	}
		
	public int compareTo(TorcsNeatAI o) {
		// TODO Auto-generated method stub
		return fitness.compareTo(o.fitness);	
	}

	public Activator getActivator(){	
		return a;
	}
	

}
