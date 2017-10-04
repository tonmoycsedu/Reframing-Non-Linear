package reframing;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author Mazid
 */
public class Quadratic_new {

    Instances train, test;
    Classifier model;
    int num;
    
    public double[] findOptimumAlpha( double p, double tmpMeanAbsoluteError, int idx ) throws Exception
    {
        int count = 0;
        double[] results = new double[2];
        double newAlpha = 1, newMeanAbsoluteError = 0;
        Instances shiftedTestSet = new Instances(test);
        Evaluation eval = new Evaluation(train);
        do {
            newAlpha += p;
            newMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                double value = test.instance(i).value(idx);
                shiftedTestSet.instance(i).setValue(idx, newAlpha * value * value );
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedTestSet);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= newMeanAbsoluteError); // continue if new result is better than older
        results[0] = newAlpha;
        results[1] = newMeanAbsoluteError;
        return results;
        
    }
    
    public double[] findOptimumBeta( double p, double alpha, double tmpMeanAbsoluteError, int idx ) throws Exception
    {
        int count = 0;
        double[] results = new double[2];
        double newBeta = 1, newMeanAbsoluteError = 0;
        Instances shiftedTestSet = new Instances(test);
        Evaluation eval = new Evaluation(train);
        do {
            newBeta += p;
            newMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                double value = test.instance(i).value(idx);
                shiftedTestSet.instance(i).setValue(idx, alpha * value * value + newBeta * value );
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedTestSet);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= newMeanAbsoluteError); // continue if new result is better than older
        results[0] = newBeta;
        results[1] = newMeanAbsoluteError;
        return results;
        
    }
    
    public double[] findOptimumGamma( double p, double alpha, double beta, double tmpMeanAbsoluteError, int idx ) throws Exception
    {
        int count = 0;
        double[] results = new double[2];
        double newGamma = 0, newMeanAbsoluteError = 0;
        Instances shiftedTestSet = new Instances(test);
        Evaluation eval = new Evaluation(train);
        do {
            newGamma += p;
            newMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                double value = test.instance(i).value(idx);
                shiftedTestSet.instance(i).setValue(idx, alpha * value * value + beta * value + newGamma );
            }
            // evaluate shifted test data
            eval.evaluateModel(model, shiftedTestSet);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= newMeanAbsoluteError); // continue if new result is better than older
        results[0] = newGamma;
        results[1] = newMeanAbsoluteError;
        return results;
        
    }

    public void selectAlphaBetaGama(int idx) throws Exception {
        double p = 0.1;
        double alpha = 1.0, beta =  1.0, gamma = 0;

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        double meanAbsoluteError = eval.meanAbsoluteError();
        
        // decrease alpha for geting better result
        double[] negAlphaResult = findOptimumAlpha(-0.1, meanAbsoluteError, idx);

        // increase alpha for geting better result
        double[] posAlphaResult = findOptimumAlpha(0.1, meanAbsoluteError, idx);

        // select best alpha
        if(negAlphaResult[1] < posAlphaResult[1] && negAlphaResult[1] < meanAbsoluteError) {
            alpha = negAlphaResult[0] + p;
            meanAbsoluteError = negAlphaResult[1];
        }
        else if(posAlphaResult[1] < negAlphaResult[1] && posAlphaResult[1] < meanAbsoluteError) {
            alpha = posAlphaResult[0] - p;
            meanAbsoluteError = posAlphaResult[1];
        }
       
        // now same procedure for beta      
        // decrease beta for geting better result
        double[] negBetaResult = findOptimumBeta(-0.1, alpha, meanAbsoluteError, idx);
        
        // increase beta for geting better result
        double[] posBetaResult = findOptimumBeta(0.1, alpha, meanAbsoluteError, idx);

        // select best beta
        if(negBetaResult[1] < posBetaResult[1] && negBetaResult[1] < meanAbsoluteError) {
            beta = negBetaResult[0] + p;
            meanAbsoluteError = negBetaResult[1];
        }
        else if(posBetaResult[1] < negBetaResult[1] && posBetaResult[1] < meanAbsoluteError) {
            beta = posBetaResult[0] - p;
            meanAbsoluteError = posBetaResult[1];
        }
        
         // now same procedure for gamma      
        // decrease gamma for geting better result
        double[] negGamaResult = findOptimumGamma(-0.1, alpha, beta, meanAbsoluteError, idx);

        // increase gamma for geting better result
        double[] posGammaResult = findOptimumGamma(0.1, alpha, beta, meanAbsoluteError, idx);

        // select best beta
        if(negGamaResult[1] < posGammaResult[1] && negGamaResult[1] < meanAbsoluteError) {
            gamma = negGamaResult[0] + p;
            meanAbsoluteError = negGamaResult[1];
        }
        else if(posGammaResult[1] < negGamaResult[1] && posGammaResult[1] < meanAbsoluteError) {
            gamma = posGammaResult[0] - p;
            meanAbsoluteError = posGammaResult[1];
        }
       
        // now shift dataset using learned alpha beta
        for(int i=0; i<num; i++) {
            double value = this.test.instance(i).value(idx);
            //if(i==0) System.out.println("before change: " + this.test.instance(0));
            this.test.instance(i).setValue(idx, alpha*value*value+beta*value+gamma );
        }
        //System.out.println("Alpha=" + alpha + " beta=" + beta);
        //System.out.println("after change: " + this.test.instance(0));
    }

    public void hillClimbing() throws Exception {
        int numOfAttributes = test.numAttributes() - 1; // excluding class attribute
        Utility ut = new Utility();
        // find optimum alpha beta for each attribute
        for(int i=0; i<numOfAttributes; i++) {
            selectAlphaBetaGama(i);
        }
        // now evaluate the shifted test data set
        Evaluation eval = new Evaluation(this.train);
        eval.evaluateModel(this.model, this.test);
        System.out.println("Accuracy for Non linear Shift: "+ut.cal_accuracy(eval.correct(), eval.incorrect()));
        //System.out.println(eval.toSummaryString("\nResults for non-linear shifted dataset\nShifted " + num + " data\n=================\n", false));
        //System.out.println(eval.);
        //System.out.println("Precision: " + eval.precision(this.test.classIndex()) + "\nRecall: " + eval.recall(this.test.classIndex()));
    }

    public void reframing(Instances train, Instances test, Classifier model, int num) throws Exception
    {
        this.train = new Instances(train);
        this.test = new Instances(test);
        this.model = model;
        this.num = num;

        // call hillclimbing
        hillClimbing();
    }
}
