/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reframing;

/**
 *
 * @author user
 */
public class Utility {
    
    public double cal_accuracy(double correct, double incorrect)
    {
        double accuracy = (correct)/(correct+incorrect);
        return accuracy;
        
    }
    
}
