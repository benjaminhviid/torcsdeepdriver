package itu.bj.torcs;

import com.anji.integration.Activator;

import static org.bj.deeplearning.tools.Utils.map;
import static org.bj.deeplearning.tools.Utils.clamp;



public class TorcsNeatController extends Controller {

	
	private boolean finished = false;
	private Activator activator;

	private static final int MAX_JUMP = 100;
	private static final int TICKS_PER_SAVE = 125; // Approx. 5 seconds.

// store dist
	private double dist = 0.0;
	private double damage = 0.0;
	private Double speed = 0.0;
	private Double angle = 0.0;
	private Double dist_RL = 0.0;
	private Double dist_RR = 0.0;


	private double lastLaps = 0.0;
	private int laps = 0;

	// change in position in the last TICKS_PER_SAVE
	private double lastDiff = 10.0;
	private double lastDist = 0.0;
	private int lastTick = 0;
	private int tick = 0;

	public TorcsNeatController(Activator ac) {
		// TODO Auto-generated constructor stub
		this.activator = ac;
		dist = 0.0;
	}
	public TorcsNeatController() {
		// TODO Auto-generated constructor stub
		super();
		dist = 0.0;
		Action action = new Action ();
	}

	Integer counter = 0;
	Double distFromLane = 0.0;


    @Override
	public Action control(SensorModel sensors) {
		// TODO Auto-generated method stub
		if (!DataCollector.instance().isRecording())
			DataCollector.Resume();
		double distance = sensors.getDistanceFromStartLine();
		
		updateDiff(distance);
		storeDistance(distance);
		Action result = getOutput(activator.next(getInput(sensors)));

		result.clutch = 0;
		result.gear = automaticGear(sensors);

		// logging values
		speed  = normalizeSpeed(sensors.getSpeed());
		angle = normalizeAngle(sensors.getAngleToTrackAxis());
		Double trackPos = sensors.getTrackPosition();

		dist_RL = map(clamp(trackPos, -1, 1), -1.0, 1.0, 0.0, 1.0);
		dist_RR = map((1 - clamp(trackPos, -1, 1)), -1.0, 1.0, 0.0, 1.0);

		DataCollector.instance().UpdateTrainingValues(angle, dist_RL, dist_RR, speed);

		damage = sensors.getDamage();
		distFromLane += map((0.5 - clamp(trackPos, -1, 1)), -1.0, 1.0, 0.0, 1.0);
		counter++;

		return result;
	}

	private int automaticGear(SensorModel sensormodel){

		int gear = sensormodel.getGear();

		switch(gear){
			case 6: if(sensormodel.getRPM() < 2000){gear = 5;} break;
			case 5: if(sensormodel.getRPM() > 9000){gear = 6;}else if(sensormodel.getRPM() < 2000){gear = 4;} break;
			case 4:	if(sensormodel.getRPM() > 9000){gear = 5;}else if(sensormodel.getRPM() < 2000){gear = 3;} break;
			case 3: if(sensormodel.getRPM() > 9000){gear = 4;}else if(sensormodel.getRPM() < 2000){gear = 2;} break;
			case 2: if(sensormodel.getRPM() > 9000){gear = 3;}else if(sensormodel.getRPM() < 2000){gear = 1;} break;
			case 1: if(sensormodel.getRPM() > 9000){gear = 2;} break;
			case 0: gear = 1; break;
			case -1:  gear = 1; break;
		}
		return gear;
	}

	private void updateDiff(double dist) {
		if(tick == lastTick + TICKS_PER_SAVE){
			lastTick = tick;
			lastDiff = Math.abs(dist - lastDist);
			lastDist = dist;
		}
		tick++;
	}

	public Double getAverageDistanceToLane(){
        return distFromLane/counter.doubleValue();
    }

	private Action getOutput(double[] output) {
		Action result = new Action();
		result.accelerate = clamp(output[0],0,1);
		result.brake = clamp(output[1],0,1);
		result.steering = normalizeSteering(output[2]);

		return result;
	}

	private double[] getInput(SensorModel sensors) {
		// TODO Auto-generated method stub
		double[] result = new double[4];
		result[0] = 1.0;
		result[1] = normalizeAngle(sensors.getAngleToTrackAxis());
		result[2] = normalizeSpeed(sensors.getSpeed());
		result[3] = normalizeTrackPos(sensors.getTrackPosition());
		
		return result;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		System.out.println("resetting");
	}

	@Override
	public void shutdown() {
		DataCollector.Pause();
	}

	private double clamp(double value, double min, double max){
		return Math.max(min, Math.min(max, value));
	}

	private void storeDistance(double newFit) {
		if(newFit > dist && dist + MAX_JUMP > newFit){ // legal progress
			dist = newFit;
		}/*else if(newFit < MAX_JUMP && dist > 5*MAX_JUMP){ // new lap
			laps++;
			lastLaps += dist;
			dist = newFit;
		}*/
	}

	public Double getScore(){
		Double score = (dist * ((1 - getAverageDistanceToLane())) * (1 - normalizeDamage(damage)));
		if (score < 0.0)
				score = 0.0;
		return score;
	}

	private double normalizeTrackPos(double pos) {
		return (pos + 1.0)/2.0;
	}

	private double[] normalizeTrack(double[] track) {
		double[] result = new double[10];
		for(int i = 0; i < result.length; i++){
			result[i] = track[i*2]/200.0;
		}
		return result;
	}

	private double normalizeSpeed(double speed) {
		return clamp((speed+200)/400.0, 0.0, 1.0);
	}

    private double normalizeDamage(double dmg) { return clamp(dmg/10000.0, 0.0, 1.0); }

    private double normalizeAngle(double angle) {
		return (angle+Math.PI)/(2*Math.PI);
	}
	
	private double normalizeSteering(double steer) {
		return clamp(steer*2.0-1.0,-1,1);
	}

	

}
