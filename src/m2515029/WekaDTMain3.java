package m2515029;

import java.awt.BorderLayout;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class WekaDTMain3 {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// read data
		Instances dataTrain = DataSource.read("/home/thanh/workspace/BT-DWDM-Weka/ReutersGrain-train.arff");
		// class is the last attribute
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);

		// read data
		Instances dataTest = DataSource.read("/home/thanh/workspace/BT-DWDM-Weka/ReutersGrain-test.arff");
		// class is the last attribute
		dataTest.setClassIndex(dataTest.numAttributes() - 1);

		// initialize string to word vector
		StringToWordVector stwv = new StringToWordVector();
		stwv.setInputFormat(dataTrain);

		// initialize classifier - decision tree
		String[] options = new String[2];
		options[0] = "-U"; // unpruned tree
		options[1] = "-i";
		Classifier tree = new J48(); // new instance of tree
		tree.setOptions(options); // set the options

		// using filter classifier to apply filtered input into classifier

		FilteredClassifier cls = new FilteredClassifier();
		cls.setFilter(stwv);
		cls.setClassifier(tree);
		cls.buildClassifier(dataTrain);

		// perform cross-validation
		Evaluation eval = new Evaluation(dataTrain);
		eval.evaluateModel(cls, dataTest);

		// output evaluation
		System.out.println();
		System.out.println("Classifier: " + tree.getClass().getName() + " " + Utils.joinOptions(tree.getOptions()));
		System.out.println("Dataset: " + dataTrain.relationName());
		System.out.println();

		System.out.println(eval.toMatrixString("=== Matrix ==="));
		System.out.println(eval.toClassDetailsString("=== Class detail ==="));

		// generate curve
		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;
		Instances result = tc.getCurve(eval.predictions(), classIndex);

		// plot curve
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		vmc.setROCString("(Area under ROC = " + Utils.doubleToString(ThresholdCurve.getROCArea(result), 4) + ")");
		vmc.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		// specify which points are connected
		boolean[] cp = new boolean[result.numInstances()];
		for (int n = 1; n < cp.length; n++)
			cp[n] = true;
		tempd.setConnectPoints(cp);
		// add plot
		vmc.addPlot(tempd);

		// display curve
		String plotName = vmc.getName();
		final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Visualize: " + plotName);
		jf.setSize(500, 400);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(vmc, BorderLayout.CENTER);
		jf.addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(java.awt.event.WindowEvent e) {
				jf.dispose();
			}
		});
		jf.setVisible(true);
	}
}
