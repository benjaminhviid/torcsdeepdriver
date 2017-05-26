package itu.bj.torcs;

/**
 * Created by IntelliJ IDEA.
 * User: Administrator
 * Date: Mar 4, 2008
 * Time: 12:18:47 PM
 */
public interface SensorModel {

    // basic information about your car and the track (you probably should take care of these somehow)

    double getSpeed();

    double getAngleToTrackAxis();

    double[] getTrackEdgeSensors();
    
    double[] getFocusSensors();//ML

    double getTrackPosition();

    int getGear();

    // basic information about other cars (only useful for multi-car races)

    double[] getOpponentSensors();

    int getRacePosition();

    // additional information (use if you need)

    double getLateralSpeed();
    

    double getCurrentLapTime();

    double getDamage();

    double getDistanceFromStartLine();

    double getDistanceRaced();

    double getFuelLevel();

    double getLastLapTime();

    double getRPM();

    double[] getWheelSpinVelocity();
    
    double getZSpeed();
    
    double getZ();
    
    String getMessage();
    
}
