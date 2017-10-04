package reframing;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author Mazid
 */
public class Reframing {

    Instances train, test;
    Classifier model;
    int num;

    public void selectAlphaBeta(int idx) throws Exception {
        double p = 0.1;
        double alpha = 1.0;
        double negAlpha = alpha, posAlpha = alpha;
        Instances shiftedNegTest = new Instances(test);
        Instances shiftedPosTest = new Instances(test);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        double meanAbsoluteError = eval.meanAbsoluteError();
        double tmpMeanAbsoluteError = meanAbsoluteError;
        double negMeanAbsoluteError;
        double posMeanAbsoluteError;

        // decrease alpha for geting better result
        int count = 0;
        do {
            negAlpha -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                shiftedNegTest.instance(i).setValue(idx, negAlpha * test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval = new Evaluation(train);
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= negMeanAbsoluteError && count < 10); // continue if new result is better than older

        // increase alpha for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        count = 0;
        do {
            posAlpha += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                shiftedPosTest.instance(i).setValue(idx, posAlpha * test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval = new Evaluation(train);
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= posMeanAbsoluteError && count < 10); // continue if new result is better than older

        // select best alpha
        if(negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            alpha = negAlpha + p;
            meanAbsoluteError = negMeanAbsoluteError;
            //test = shiftedNegTest;
        }
        else if(posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            alpha = posAlpha - p;
            meanAbsoluteError = posMeanAbsoluteError;
            //test = shiftedPosTest;
        }

        // now same procedure for beta
        negMeanAbsoluteError = meanAbsoluteError;
        posMeanAbsoluteError = meanAbsoluteError;
        tmpMeanAbsoluteError = meanAbsoluteError;

        shiftedNegTest = new Instances(test);
        shiftedPosTest = new Instances(test);

        double beta = 0.0;
        double negBeta = beta, posBeta = beta;

        // decrease beta for geting better result
        count = 0;
        do {
            negBeta -= p;
            negMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                shiftedNegTest.instance(i).setValue(idx, negBeta + test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval = new Evaluation(train);
            eval.evaluateModel(model, shiftedNegTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= negMeanAbsoluteError && count < 10); // continue if new result is better than older

        // increase beta for geting better result
        tmpMeanAbsoluteError = meanAbsoluteError;
        count = 0;
        do {
            posBeta += p;
            posMeanAbsoluteError = tmpMeanAbsoluteError; // save better meanAbsoluteError
            for(int i=0; i<num; i++) {
                shiftedPosTest.instance(i).setValue(idx, posBeta + test.instance(i).value(idx));
            }
            // evaluate shifted test data
            eval = new Evaluation(train);
            eval.evaluateModel(model, shiftedPosTest);
            tmpMeanAbsoluteError = eval.meanAbsoluteError();
            count++;
        }while(tmpMeanAbsoluteError <= posMeanAbsoluteError && count < 10); // continue if new result is better than older

        // select best beta
        if(negMeanAbsoluteError < posMeanAbsoluteError && negMeanAbsoluteError < meanAbsoluteError) {
            beta = negBeta + p;
            meanAbsoluteError = negMeanAbsoluteError;
            //test = shiftedNegTest;
        }
        else if(posMeanAbsoluteError < negMeanAbsoluteError && posMeanAbsoluteError < meanAbsoluteError) {
            beta = posBeta - p;
            meanAbsoluteError = posMeanAbsoluteError;
            //test = shiftedPosTest;
        }

        // now shift dataset using learned alpha beta
        for(int i=0; i<num; i++) {
            //if(i==0) System.out.println("before change: " + this.test.instance(0));
            this.test.instance(i).setValue(idx, this.test.instance(i).value(idx)*alpha+beta );
        }
        //System.out.println("Alpha=" + alpha + " beta=" + beta);
        //System.out.println("after change: " + this.test.instance(0));
    }

    public void hillClimbing() throws Exception {
        int numOfAttributes = test.numAttributes() - 1; // excluding class attribute
        Utility ut = new Utility();
        // find optimum alpha beta for each attribute
        for(int i=0; i<numOfAttributes; i++) {
            selectAlphaBeta(i);
        }
        // now evaluate the shifted test data set
        Evaluation eval = new Evaluation(this.train);
        eval.evaluateModel(this.model, this.test);
        System.out.println("Accuracy for linear Shift: "+ut.cal_accuracy(eval.correct(), eval.incorrect()));
        //System.out.println(eval.toSummaryString("\nResults for linear shifted dataset\nShifted " + num + " data\n=================\n", false));
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
