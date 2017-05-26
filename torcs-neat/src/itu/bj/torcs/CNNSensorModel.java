package itu.bj.torcs;

import static org.bj.deeplearning.tools.Utils.clamp;
import static org.bj.deeplearning.tools.Utils.map;

/**
 * Created by benjaminhviid on 5/2/17.
 */
public class CNNSensorModel {

    private static CNNSensorModel instance = null;
    private CNNSensorModel() {
        // Exists only to defeat instantiation.
    }

    public static CNNSensorModel getInstance() {
        if(instance == null) {
            instance = new CNNSensorModel();
        }
        return instance;
    }

    private double angleToTrackAxis = 0.0;
    private double trackPosition = 0.0;

    public void setValues(double _angleToTrackAxis, double _trackPos){
        this.angleToTrackAxis = revertNormalizedAngleToTrackAxis(_angleToTrackAxis);
        this.trackPosition = revertNormalizedTrackPosition(_trackPos);
    }

    public double getAngleToTrackAxis(){
        return angleToTrackAxis;
    }

    public double getTrackPosition(){
        return trackPosition;
    }


    private double revertNormalizedAngleToTrackAxis(double angleToTrackAxis){
        return map(angleToTrackAxis,0.0, 1.0,  -Math.PI, Math.PI);

    }

    private double revertNormalizedTrackPosition(double trackPosition){
        return map(trackPosition, 0.0, 1, -2.0, 2.0);

    }



}
