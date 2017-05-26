/**
 * 
 */
package itu.bj.torcs;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.Buffer;
import java.util.StringTokenizer;

import itu.bj.torcs.Controller.Stage;
import org.bj.deeplearning.executables.Evaluator;
import org.bj.deeplearning.tools.ImageTool;
import org.bj.deeplearning.tools.Utils;

import static org.bj.deeplearning.tools.Utils.resize;

/**
 * @author Daniele Loiacono
 * 
 */
public class Client {

	private static int UDP_TIMEOUT = 10000;
	private static int port;
	private static String host;
	private static String clientId;
	private static boolean verbose;
	private static int maxEpisodes;
	private static int maxSteps;
	private static Stage stage;
	private static String trackName;
	private static Controller driver;
	private static Double finalScore; 
	private static PrintWriter writer;
	private static boolean enableCNN = true;
	private static MyThread thread = new MyThread();


	public Client(Controller driver){
		Client.driver = driver;

	}


	/**
	 * @param args
	 *            is used to define all the options of the client.
	 *            <port:N> is used to specify the port for the connection (default is 3001)
	 *            <host:ADDRESS> is used to specify the address of the host where the server is running (default is localhost)  
	 *            <id:ClientID> is used to specify the ID of the client sent to the server (default is championship2009) 
	 *            <verbose:on> is used to set verbose mode on (default is off)
	 *            <maxEpisodes:N> is used to set the number of episodes (default is 1)
	 *            <maxSteps:N> is used to set the max number of steps for each episode (0 is default value, that means unlimited number of steps)
	 *            <stage:N> is used to set the current stage: 0 is WARMUP, 1 is QUALIFYING, 2 is RACE, others value means UNKNOWN (default is UNKNOWN)
	 *            <trackName:name> is used to set the name of current track
	 */
	public static void main(String[] args) {

		parseParameters(args);

		SocketHandler mySocket = new SocketHandler(host, port, verbose);
		String inMsg;
		finalScore = 0.0;
		//driver = load(args[0]);
		//driver = new TorcsNeatController();
		//driver = new Kitt();
        driver.setStage(stage);
		driver.setTrackName(trackName);
		Thread th = new Thread(thread);
		/* Build init string */
		float[] angles = driver.initAngles();
		String initStr = clientId + "(init";
		for (int i = 0; i < angles.length; i++) {
			initStr = initStr + " " + angles[i];
		}
		initStr = initStr + ")";

		long curEpisode = 0;
		boolean shutdownOccurred = false;


		if (enableCNN){
			th.start();
		}
		do {

			/*
			 * Client identification
			 */

			do {
				mySocket.send(initStr);
				inMsg = mySocket.receive(UDP_TIMEOUT);
			} while (inMsg == null || inMsg.indexOf("***identified***") < 0);

			/*
			 * Start to drive
			 */
			//DataCollector.instance().StartDataCollection(333);


			long currStep = 0;
			while (true) {

				/*
				 * Receives from TORCS the game state
				 */
				inMsg = mySocket.receive(UDP_TIMEOUT);

				if (inMsg != null) {

					/*
					 * Check if race is ended (shutdown)
					 */

					if (inMsg.indexOf("***shutdown***") >= 0) {
						shutdownOccurred = true;
						System.out.println("Server shutdown!");
						break;
					}

					/*
					 * Check if race is restarted
					 */
					if (inMsg.indexOf("***restart***") >= 0) {
						driver.reset();
						if (verbose)
							System.out.println("Server restarting!");
						break;
					}

					Action action = new Action();
					//System.out.println(currStep + " | " + maxSteps);
					
					try{
						PrintWriter writer = new PrintWriter("botlog.txt", "UTF-8");
						writer.println(inMsg.toString());
						writer.close();
						
					} catch (Exception e){
						System.out.println(e.getMessage());
					}

					if (currStep < maxSteps || maxSteps == 0){


						action = driver.control(new MessageBasedSensorModel(
								inMsg));
						}

					currStep++;
					mySocket.send(action.toString());

				}
				else
					System.out.println("Server did not respond within the timeout");
			}

		} while (++curEpisode < maxEpisodes && !shutdownOccurred);

		/*
		 * Shutdown the controller
		 */
		driver.shutdown();
		mySocket.close();
		System.out.println("Client shutdown.");
		System.out.println("Bye, bye!");

	}

	private static void parseParameters(String[] args) {
		/*
		 * Set default values for the options
		 */
		
		
		port = 3001;
		host = "localhost";
		clientId = "SCR";
		verbose = false;
		maxEpisodes = 1;
		maxSteps = 0;
		stage = Stage.UNKNOWN;
		trackName = "unknown";
		
		for (int i = 1; i < args.length; i++) {

			StringTokenizer st = new StringTokenizer(args[i], ":");
			String entity = st.nextToken();
			String value = st.nextToken();
			if (entity.equals("port")) {
				port = Integer.parseInt(value);
			}
			if (entity.equals("host")) {
				host = value;
			}
			if (entity.equals("id")) {
				clientId = value;
			}
			if (entity.equals("verbose")) {
				if (value.equals("on"))
					verbose = true;
				else if (value.equals(false))
					verbose = false;
				else {
					System.out.println(entity + ":" + value
							+ " is not a valid option");
					System.exit(0);
				}
			}
			if (entity.equals("id")) {
				clientId = value;
			}
			if (entity.equals("stage")) {
				stage = Stage.fromInt(Integer.parseInt(value));
			}
			if (entity.equals("trackName")) {
				trackName = value;
			}
			if (entity.equals("maxEpisodes")) {
				maxEpisodes = Integer.parseInt(value);
				if (maxEpisodes <= 0) {
					System.out.println(entity + ":" + value
							+ " is not a valid option");
					System.exit(0);
				}
			}
			if (entity.equals("maxSteps")) {
				maxSteps = Integer.parseInt(value);
				if (maxSteps < 0) {
					System.out.println(entity + ":" + value
							+ " is not a valid option");
					System.exit(0);
				}
			}
		}
		System.out.println(port + " " + host + " " + clientId + " " + verbose + " " + maxEpisodes + " "+ 
				
				maxSteps + " " + stage  + " " + trackName);

	}
	
	public static void setDriver(Controller bot){
		driver = bot;
	}
	public static Controller getDriver(){
		return driver;
	}

	private static Controller load(String name) {
		Controller controller=null;
		try {
			controller = (Controller) Class.forName(name)
					.newInstance();
		} catch (ClassNotFoundException e) {
			System.out.println(name	+ " is not a class name");
			System.exit(0);
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return controller;
	}
	
	public static Double getFinalScore(){
		return ((TorcsNeatController)driver).getScore();
	}
}

class MyThread implements Runnable {

	public void run()
	{
		int counter = 0;
		try {
			while (true) {
				counter++;
				byte[] pixeldata = ImageTool.bufferedImageToByteArray(resize(Utils.getScreenshot(480, 640), 80, 60));
				double[] out = Evaluator.getOutput(pixeldata);
				System.out.println(out[0] + "       " + out[1]);
				CNNSensorModel.getInstance().setValues(out[0], out[1]);
				//ImageTool.printColoredPngImage(pixeldata, 80, new File("img" + counter + ".png"));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}