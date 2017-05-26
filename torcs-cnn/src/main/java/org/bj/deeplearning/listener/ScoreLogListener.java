package org.bj.deeplearning.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class ScoreLogListener implements IterationListener {
    private final int printIterations;
    private static final Logger log = LoggerFactory.getLogger(ScoreLogListener.class);
    private boolean invoked = false;
    private long iterCount = 0;
    PrintWriter pw;
    boolean modelConfigLogged = false;
    String name;
    /**
     * @param printIterations    frequency with which to print scores (i.e., every printIterations parameter updates)
     */
    public ScoreLogListener(int printIterations, String name) {
    	if(printIterations <= 0) {
            printIterations = 1;
        }
        this.name = name;
        this.printIterations = printIterations;
        try {
            File file = new File(name+".txt");
            pw= new PrintWriter(file);
            pw.println(name + "(frequency: "+ printIterations +" iterations)\n");
            pw.close();
            System.out.println("File with name "+name+".txt created");
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    /** Default constructor printing every 10 iterations */
    public ScoreLogListener() {
    	this(10, "scorelog");
    }

    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }

    @Override
    public void iterationDone(Model model, int iteration) {
    	long now = System.currentTimeMillis();

        if(iterCount++ % printIterations == 0) {
            invoke();

            FileWriter fw= null;
            try {
                fw = new FileWriter(name+".txt", true);
            } catch (IOException e) {
                e.printStackTrace();
            }
            PrintWriter p= new PrintWriter(fw);
            if (!modelConfigLogged){
                modelConfigLogged = true;

                p.println(model.conf().toJson() + "\n");
                p.close();
            }
            p.println(Double.toString(model.score()));
            p.close();
            //System.out.println("Time of iteration " + iterCount + " is " + (now-timeOfLast) + " ms");
        }
    }
}
