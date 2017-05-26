package itu.bj.torcs;

/**
 * Created by benjaminhviid on 5/2/17.
 */
public class CNNBot extends Controller {


    /* Gear Changing Constants*/
    final int[]  gearUp={5000,6000,6000,6500,7000,0};
    final int[]  gearDown={0,2500,3000,3000,3500,3500};

    /* Stuck constants*/
    final int  stuckTime = 25;
    final float  stuckAngle = (float) 0.523598775; //PI/6

    /* Accel and Brake Constants*/
    final float maxSpeedDist=70;
    final float maxSpeed=80;
    final float sin5 = (float) 0.08716;
    final float cos5 = (float) 0.99619;

    /* Steering constants*/
    final float steerLock=(float) 0.366519;
    final float steerSensitivityOffset=(float) 80.0;
    final float wheelSensitivityCoeff=1;

    /* ABS Filter Constants */
    final float wheelRadius[]={(float) 0.3306,(float) 0.3306,(float) 0.3276,(float) 0.3276};
    final float absSlip=(float) 2.0;
    final float absRange=(float) 3.0;
    final float absMinSpeed=(float) 3.0;

    /* Clutching Constants */
    final float clutchMax=(float) 0.5;
    final float clutchDelta=(float) 0.05;
    final float clutchRange=(float) 0.82;
    final float	clutchDeltaTime=(float) 0.02;
    final float clutchDeltaRaced=10;
    final float clutchDec=(float) 0.01;
    final float clutchMaxModifier=(float) 1.3;
    final float clutchMaxTime=(float) 1.5;

    private int stuck=0;

    // current clutch
    private float clutch=0;

    private double targetSpeed = 0;

    private int getGear(SensorModel sensors){
        int gear = sensors.getGear();
        double rpm  = sensors.getRPM();

        // if gear is 0 (N) or -1 (R) just return 1
        if (gear < 1)
            return 1;
        // check if the RPM value of car is greater than the one suggested
        // to shift up the gear from the current one
        if (gear <6 && rpm >= gearUp[gear-1])
            return gear + 1;
        else
            // check if the RPM value of car is lower than the one suggested
            // to shift down the gear from the current one
            if (gear > 1 && rpm <= gearDown[gear-1])
                return gear - 1;
            else // otherwhise keep current gear
                return gear;
    }

    private float getSteer(SensorModel sensorModel, CNNSensorModel cnnSensorModel){
        // steering angle is compute by correcting the actual car angle w.r.t. to track
        // axis [sensors.getAngle()] and to adjust car position w.r.t to middle of track [sensors.getTrackPos()*0.5]
        float targetAngle=(float) (cnnSensorModel.getAngleToTrackAxis()-cnnSensorModel.getTrackPosition() * 0.5);
        // at high speed reduce the steering command to avoid loosing the control
        System.out.println("Expected angle: " + sensorModel.getAngleToTrackAxis() + " | cnn angle: " + cnnSensorModel.getAngleToTrackAxis());
        System.out.println("Expected trackPos: " + sensorModel.getTrackPosition() + " | cnn trackPos: " + cnnSensorModel.getTrackPosition());
        System.out.println("");

        if (sensorModel.getSpeed() > steerSensitivityOffset)
            return (float) (targetAngle/(steerLock*(sensorModel.getSpeed()-steerSensitivityOffset)*wheelSensitivityCoeff));
        else
            return (targetAngle)/steerLock;
    }

    private float getAccel(SensorModel sensors, CNNSensorModel cnnSensors)
    {
        // checks if car is out of track
        if (sensors.getTrackPosition() < 1 && sensors.getTrackPosition() > -1)
        {
            // reading of sensor at +5 degree w.r.t. car axis
            float rxSensor=(float) sensors.getTrackEdgeSensors()[10];
            // reading of sensor parallel to car axis
            float sensorsensor=(float) sensors.getTrackEdgeSensors()[9];
            // reading of sensor at -5 degree w.r.t. car axis
            float sxSensor=(float) sensors.getTrackEdgeSensors()[8];

            float targetSpeed;

            // track is straight and enough far from a turn so goes to max speed
            if (sensorsensor>maxSpeedDist || (sensorsensor>=rxSensor && sensorsensor >= sxSensor))
                targetSpeed = maxSpeed;
            else
            {
                // approaching a turn on right
                if(rxSensor>sxSensor)
                {
                    // computing approximately the "angle" of turn
                    float h = sensorsensor*sin5;
                    float b = rxSensor - sensorsensor*cos5;
                    float sinAngle = b*b/(h*h+b*b);
                    // estimate the target speed depending on turn and on how close it is
                    targetSpeed = maxSpeed*(sensorsensor*sinAngle/maxSpeedDist);
                }
                // approaching a turn on left
                else
                {
                    // computing approximately the "angle" of turn
                    float h = sensorsensor*sin5;
                    float b = sxSensor - sensorsensor*cos5;
                    float sinAngle = b*b/(h*h+b*b);
                    // estimate the target speed depending on turn and on how close it is
                    targetSpeed = maxSpeed*(sensorsensor*sinAngle/maxSpeedDist);
                }

            }

            // accel/brake command is exponentially scaled w.r.t. the difference between target speed and current one
            return (float) (2/(1+Math.exp(sensors.getSpeed() - targetSpeed)) - 1);
        }
        else
            return (float) 0.3; // when out of track returns a moderate acceleration command

    }


    @Override
    public Action control(SensorModel sensors) {

        Action action = new Action ();
        float accel_and_brake = getAccel(sensors, CNNSensorModel.getInstance());
        // compute gear
        int gear = getGear(sensors);
        // compute steering
        float steer = getSteer(sensors, CNNSensorModel.getInstance());

        // normalize steering
        if (steer < -1)
            steer = -1;
        if (steer > 1)
            steer = 1;

        // set accel and brake from the joint accel/brake command
        float accel;
        float brake;
        if (accel_and_brake>0)
        {
            accel = accel_and_brake;
            brake = 0;
        }
        else
        {
            accel = 0;
            brake = filterABS(sensors,-accel_and_brake);

        }

        action.steering = steer;
        action.brake = brake;
        action.accelerate = accel;

        action.gear = getGear(sensors);
        return action;
    }

    public void reset() {
        System.out.println("Restarting the race!");

    }
    private float filterABS(SensorModel sensors,float brake){
        // convert speed to m/s
        float speed = (float) (sensors.getSpeed() / 3.6);
        // when spedd lower than min speed for abs do nothing
        if (speed < absMinSpeed)
            return brake;

        // compute the speed of wheels in m/s
        float slip = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            slip += sensors.getWheelSpinVelocity()[i] * wheelRadius[i];
        }
        // slip is the difference between actual speed of car and average speed of wheels
        slip = speed - slip/4.0f;
        // when slip too high applu ABS
        if (slip > absSlip)
        {
            brake = brake - (slip - absSlip)/absRange;
        }

        // check brake is not negative, otherwise set it to zero
        if (brake<0)
            return 0;
        else
            return brake;
    }


      public void shutdown() {
        System.out.println("Bye bye!");
    }
}
