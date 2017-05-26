package org.bj.deeplearning.tools;

import org.bj.deeplearning.dataobjects.TrainingDataHandler;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 * Created by benjaminhviid on 15/04/2017.
 */
public class Utils {

    private Utils(){}

    private static int offsetX = 64;
    private static int offsetY = 52;
    private static int width;
    private static int height;
    private static Properties projectProperties = PropertiesReader.getProjectProperties();
    public static double map(double s, double a1, double a2, double b1, double b2)
    {
        return b1 + (s-a1)*(b2-b1)/(a2-a1);
    }

    public static double clamp (double value, double min, double max){
        return Math.max(min, Math.min(max, value));
    }


    public static  BufferedImage getScreenshot(int height, int width){
        //width = Integer.parseInt(projectProperties.getProperty("training.image.width"));
        //height = Integer.parseInt(projectProperties.getProperty("training.image.height"));
        try {
            BufferedImage image = new Robot().createScreenCapture(new Rectangle(offsetX, offsetY, width, height));
            BufferedImage resized = resize(image, width, height);

            /*String screenshotFolder = "/screenshots";
            String screenshotDir = newDir + screenshotFolder;

            if (!(new File(screenshotDir).exists())){
                new File(screenshotDir).mkdirs();
            }*/
            //ImageIO.write(resized, "jpg", new File(screenshotDir + "/screenshot"+screenCount+".jpg"));

            return resized;

        } catch (HeadlessException | AWTException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return null;
    }



    public static BufferedImage resize(BufferedImage img, int newW, int newH) {

        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }

}
