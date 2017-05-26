package org.bj.deeplearning.dataobjects;

import org.bj.deeplearning.tools.INDArrayTool;
import org.bj.deeplearning.tools.ImageTool;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

import static org.bj.deeplearning.tools.Utils.clamp;
import static org.bj.deeplearning.tools.Utils.map;


public class TrainingData {

	private int height, width, id;
	private byte[] pixelData;
	private double[] features;
	private double angle, rangeLeft, rangeForward, rangeRight, trackPos;
	private double[] d_pixeldata;
	static TrainingDataType type;

private int index = 0; // skip speed


    public static int getFeatureCount() {
   		return TrainingDataHandler.getNumberOfGroundTruths();
	}

	public TrainingData(int id, TrainingDataType trainingDataType, RunType runType) {
		int _id = id;
		if (_id == 0)
			_id = 1;

		String[] sample = new String[0];
		try {
			sample = TrainingDataHandler.getSample(_id, TrainingDataHandler.runType);
		} catch (IOException e) {
			e.printStackTrace();
		}
		TrainingData.type = trainingDataType;

		if (trainingDataType == TrainingDataType.ANGULAR) {
			angle = Double.parseDouble(sample[index++]);
			angle = clamp(angle, -Math.PI, Math.PI);
			angle = map(angle, -Math.PI, Math.PI, 0.0, 1.0);

			index+=5; // skip ranges
		}
		else if (trainingDataType == TrainingDataType.RANGE){
			index ++; // skip range 0

			rangeLeft = Double.parseDouble(sample[index++]);
			rangeLeft = map(rangeLeft, 0, 200, 0.0, 1.0);

			rangeForward = Double.parseDouble(sample[index++]);
			rangeForward = map(rangeForward, 0, 200, 0.0, 1.0);

			rangeRight = Double.parseDouble(sample[index++]);
			rangeRight = map(rangeRight, 0, 200, 0.0, 1.0);
			index++; // skip range 18
		}

		// when clamping trackpos we use -2, 2, as we give 2 x track width as track boundary in each side
		double trackPosition = Double.parseDouble(sample[index++]);
		trackPos = map(clamp(trackPosition, -2, 2), -2.0, 2.0, 0.0, 1.0);

		height = TrainingDataHandler.getHeight();
		width = TrainingDataHandler.getWidth();
		this.id = Integer.parseInt(sample[index]);


		if (runType == RunType.TRAINING) {
			pixelData = ImageTool.bufferedImageToByteArray(TrainingDataHandler.SCREENSHOTS_PATH + "screenshot" + id + ".jpg");

			try {
				INDArray indArray = TrainingDataHandler.imgLoader.asMatrix(new File(TrainingDataHandler.SCREENSHOTS_PATH + "screenshot" + id + ".jpg"));
				d_pixeldata = INDArrayTool.toFlatDoubleArray(indArray);

			} catch (IOException e) {
				e.printStackTrace();
			}

		} else
			pixelData = ImageTool.bufferedImageToByteArray(TrainingDataHandler.TEST_SCREENSHOTS_PATH + "screenshot" + id + ".jpg");


		features = calculateFeatures();

	}
	// used in FileSystem
	@Deprecated
	public TrainingData(double angle, double speed, double trackPos, int height, int width, int id, byte[] pixelData) {

		this.angle = angle;
		this.trackPos = trackPos;
		this.height = height;
		this.width = width;
		this.id = id;
		this.pixelData = pixelData;
		try {
			INDArray indArray = TrainingDataHandler.imgLoader.asMatrix(new File(TrainingDataHandler.SCREENSHOTS_PATH + "screenshot" + id + ".jpg"));
			d_pixeldata = INDArrayTool.toFlatDoubleArray(indArray);
		} catch (IOException e) {
			e.printStackTrace();
		}

		features = calculateFeatures();
	}

	private double[] calculateFeatures() {
		if (type == TrainingDataType.ANGULAR){
			double[] result = new double[2];
			result[0] = angle;
			result[1] = trackPos;
			return result;
		}
		else if (type == TrainingDataType.RANGE ) {

			double[] result = new double[4];
			result[0] = rangeLeft;
			result[1] = rangeForward;
			result[2] = rangeRight;
			result[3] = trackPos;
			return result;
		}
		else {
			return new double[]{};
		}


	}

	public double[] getFeatures() {
		return features;
	}

	public int getHeight() {
		return height;
	}

	public int getWidth() {
		return width;
	}

	public byte[] getPixelData() {
		return pixelData;
	}

	public int getId() {
		return id;
	}

	public double getAngle() {
		return angle;
	}

	public double getTrackPos(){return trackPos; }

	public double getRangeLeft() {
		return rangeLeft;
	}

	public double getRangeForward() {
		return rangeForward;
	}

	public double getRangeRight() {
		return rangeRight;
	}

	public double[] getD_pixeldata(){ return d_pixeldata;}

	public TrainingDataType getType(){ return type; }

}
