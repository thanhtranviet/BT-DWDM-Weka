package m2515029;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaKNNMain {

	public static void main(String[] args) throws Exception {
		// read data
		Instances data = DataSource.read("/home/thanh/workspace/BT-DWDM-Weka/credit-g.arff");
		// class is the last attribute
		data.setClassIndex(data.numAttributes() - 1);

		// manually create train and test set
		int trainSize = (int) Math.round(data.numInstances() * 2 / 3);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);

		/*
		// create classifier
		IBk knn = new IBk();
		// let IBk auto select the best k value from 0 to 20
		String[] options = new String[4];
		options[0] = "-K";
		options[1] = "20";
		options[2] = "-X";
		options[3] = "-E";
		knn.setOptions(options);
		 */
		// create evaluation
		

		double maxAUC = 0;
		int maxk = 1;
		IBk bestknn = new IBk();
		
		
		for(int i=1;i< train.numInstances();i++){
			// train
			IBk knn = new IBk(i);
			knn.buildClassifier(train);
			// validate with test data
			Evaluation eval = new Evaluation(data);	
			eval.evaluateModel(knn, test);
			
			if(eval.weightedAreaUnderROC() > maxAUC){
				maxAUC = eval.weightedAreaUnderROC();
				maxk = i;
				bestknn = (IBk)Classifier.makeCopy(knn);
			}
		}
		
		//compare to DTTree
		String[] options = new String[2];
		options[0] = "-U"; // unpruned tree
		options[1] = "-i";
		Classifier tree = new J48(); // new instance of tree
		tree.setOptions(options);
		tree.buildClassifier(train);
		Evaluation treeeval = new Evaluation(data);
		treeeval.evaluateModel(tree, test);
		
		double treeAUC = treeeval.weightedAreaUnderROC();
		
		// output result
		Evaluation besteval = new Evaluation(data);
		besteval.evaluateModel(bestknn, test);
		System.out.println();
		System.out.println("Classifier: " + bestknn.getClass().getName() + " " + Utils.joinOptions(bestknn.getOptions()));
		System.out.println("Dataset: " + data.relationName());
		System.out.println("Best k founded: " + bestknn.getKNN());
		System.out.println("Best KNN AUC: " + maxAUC);
		System.out.println("DTTree AUC: " + treeAUC);
		System.out.println("KNN correct predict: " + besteval.correct());
		System.out.println("DTTree correct predict: " + treeeval.correct());
		System.out.println();
		
		System.out.println(besteval.toMatrixString("=== Matrix ==="));
		System.out.println(besteval.toClassDetailsString("=== Class detail ==="));
		System.out.println(besteval.toSummaryString("=== Summary ===", false));
		

	}

}
