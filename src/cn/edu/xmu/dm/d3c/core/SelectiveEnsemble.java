package cn.edu.xmu.dm.d3c.core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.collections.CollectionUtils;

import cn.edu.xmu.dm.d3c.threads.ClassifierIndex;
import cn.edu.xmu.dm.d3c.threads.IndependentTrainThread;
import cn.edu.xmu.dm.d3c.threads.ThreadListener;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;

public class SelectiveEnsemble {

	// 交叉模型
	public List<Integer> CrossValidationModel(Instances input,
			Classifier classifier, int numfolds, List<Double> correctRateArray,
			Random random, int index, List<ClassifierIndex> qqs) {
		//
		List<Integer> ClassifyResult = new ArrayList();
		//
		try {
			//
			input = new Instances(input);
			//
			int i, j;
			//
			double correctRate;
			//
			Evaluation eval = new Evaluation(input);
			//
			for (i = 0; i < numfolds; i++) {
				// 获得训练集
				Instances train = input.trainCV(numfolds, i, random);
				// 获得测试集
				Instances test = input.testCV(numfolds, i);
				//
				eval.setPriors(train);
				//
				// Classifier copiedClassifier =Classifier.makeCopy(classifier);
				Classifier copiedClassifier = (Classifier) new SerializedObject(
						classifier).getObject();
				//
				copiedClassifier.buildClassifier(train);
				//
				double[] predictions = new double[test.numInstances()];
				
				predictions = eval.evaluateModel(copiedClassifier, test);
				//
				Instance testInst;
				//
				for (j = 0; j < test.numInstances(); j++) {
					//
					testInst = test.instance(j);
					//
					if (testInst.classValue() != predictions[j]) {
						ClassifyResult.add(0);
					} else {
						ClassifyResult.add(1);
					}
				}
			}
			correctRate = 1 - eval.errorRate();
			//
			DecimalFormat df = new DecimalFormat("0.00000");
			correctRateArray.add(correctRate);
//			System.out.print("分类器" + index +": "+df.format(correctRate));
//			System.out.println();
			qqs.add(new ClassifierIndex(index, correctRate));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ClassifyResult;
	}

	// 计算组合数
	public int CalculateCombination(int n, int m) {
		int factorial1 = 1, factorial2 = 1;
		int result = 0;
		int i;
		//
		if (n > m) {
			for (i = n; i >= (n - 1); i--) {
				factorial1 = factorial1 * i;
			}
			for (i = 1; i <= 2; i++) {
				factorial2 = factorial2 * i;
			}
			result = factorial1 / factorial2;
		} else if (n == m) {
			result = 1;
		}
		return result;
	}

	// 计算两个分类器之间的互补指数
	public double CalculateCF(List<Integer> first, List<Integer> second,
			int numInstances) {
		double CF;
		//
		List<Integer> errorUnion = new ArrayList();
		List<Integer> errorIntersect = new ArrayList();
		//
		errorUnion = (List<Integer>) CollectionUtils.union(first, second);
		errorIntersect = (List<Integer>) CollectionUtils.intersection(first,
				second);
		//
		CF = (double) (errorUnion.size() - errorIntersect.size())
				/ (double) numInstances;
		//
		// System.out.println(errorUnion);
		// System.out.println(errorIntersect);
		// System.out.println(CF);
		return CF;
	}

	// 计算整体互补指数
	public double CalculateTF(List<Integer> D, int c, double[][] CFs) {
		double TF = 0;
		double part1 = 0, part2 = 0, part3;
		int i, j;
		//
		for (i = 0; i < D.size(); i++) {
			for (j = 0; j < D.size(); j++) {
				part1 = part1 + CFs[D.get(i)][D.get(j)];
			}
		}
		//
		for (i = 0; i < D.size(); i++) {
			part2 = part2 + CFs[c][D.get(i)];
		}
		//
		part3 = 2 * (double) CalculateCombination(D.size() + 1, 2);
		// System.out.print(D.size()+1);
		//
		TF = (part1 + 2 * part2) / part3;
		// System.out.println(TF);
		return TF;
	}

	// 根据互补指数排序
	public List<Integer> Sort(List<Integer>[] classifyErrorNo,
			int numInstances, List<Double> correctRateArray) {
		int i, j;
		double[][] CFs = new double[classifyErrorNo.length][classifyErrorNo.length];
		// 计算分类器之间的互补指数
		for (i = 0; i < classifyErrorNo.length; i++) {
			for (j = 0; j < classifyErrorNo.length; j++) {
				if (i != j) {
					CFs[i][j] = CalculateCF(classifyErrorNo[i],
							classifyErrorNo[j], numInstances);
				} else if (i == j) {
					CFs[i][j] = 0;
				}
			}
		}
		//
		/*
		 * for(i=0;i<classifyErrorNo.length;i++){ for (j = 0; j <
		 * classifyErrorNo.length; j++) { System.out.print(CFs[i][j]+"	"); }
		 * System.out.println(); }
		 */
		List<Integer> T = new ArrayList();
		List<Integer> D = new ArrayList();
		//
		for (i = 0; i < classifyErrorNo.length; i++) {
			T.add(i);
		}
		// System.out.println(T);
		//
		/*
		 * D.add(T.get(0)); T.remove(0);
		 */
		double correctRate = Collections.max(correctRateArray);
		int maxNo = correctRateArray.indexOf(correctRate);
		D.add(T.get(maxNo));
		T.remove(maxNo);
		//
		double maxTF, tempTF;
		double tempMax = 0;
		double temp;

		int tempNo = 0;
		//
		DecimalFormat df = new DecimalFormat("0.00000");
		// System.out.println(ClassifierNo);

		while (T.size() != 0) {
			maxTF = Double.MIN_VALUE;
			for (i = 0; i < T.size(); i++) {
				//
				tempTF = CalculateTF(D, T.get(i), CFs);
				//
				if (tempTF > maxTF) {
					maxTF = tempTF;
					tempNo = i;
				}
			}
			//
			temp = maxTF - tempMax;
			// System.out.println(/*T.get(tempNo)+"	"+df.format(maxTF)+"	"+*/df.format(Math.abs(temp)));
			// System.out.println(T.get(tempNo)+"	"+df.format(maxTF)+"	"+df.format(temp));
			//
			tempMax = maxTF;
			//
			D.add(T.get(tempNo));
			T.remove(tempNo);
		}
		// System.out.println(tempMax);
		System.out.println("排序：" + D);
		return D;
	}

	// 单独训练
	public void IndependentTrain(Instances input, Classifier[] cfsArray,
			int numfolds, List<Integer>[] list,
			List<Integer>[] classifyErrorNo, List<Double> correctRateArray, List<ClassifierIndex> qqs)
			throws Exception {

		//
		int i, j;
		// 用于记录每种分类器单独训练时分类情况，1表示分类正确，0表示分类错误
		for (i = 0; i < cfsArray.length; i++) {
			list[i] = new ArrayList();
		}
		// 用于得到每种分类器单独训练时候错分实例的编号
		for (i = 0; i < cfsArray.length; i++) {
			classifyErrorNo[i] = new ArrayList();
		}
		// 获得list,classifyErrorNo,correctRateArray三种结构的值
		Random random = new Random();
		Instances inputR = new Instances(input);
		inputR.randomize(random);

		ThreadListener tl = new ThreadListener();//监听统计文件线程的监控线程
		for (i = 0; i < cfsArray.length; i++) {
			IndependentTrainThread itt = new IndependentTrainThread(inputR,
					cfsArray[i], numfolds, correctRateArray, random, list, i, classifyErrorNo, qqs);
			tl.array.add(itt);
			itt.start();
		}
		
		tl.start();

		while(!ThreadListener.isOver){
			
			try {//每隔一定时间监听一次各个文件统计线程
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		
		System.err.println("基分类器单独训练完成!");
	}
}


//class IndependentTrainThread extends Thread {
//	private double value;
//	private Instances input;
//	private Classifier classifier;
//	private int numfolds;
//	private List<Double> correctRateArray;
//	private Random random;
//	private List<Integer> lst;
//	private List<Integer>[] classifyErrorNo;
//	private List<Integer>[] list;
//	private int i;
//	private List<QQ> qqs;
//	
//	private long executeTime;
//	
//	private boolean isFinished;
//	
//	IndependentTrainThread(Instances input, Classifier classifier,
//			int numfolds, List<Double> correctRateArray, Random random, List<Integer>[] list, int i, List<Integer>[] classifyErrorNo, List<QQ> qqs) {
//		this.input = input;
//		this.classifier = classifier;
//		this.numfolds = numfolds;
//		this.correctRateArray = correctRateArray;
//		this.random = random;
//		this.classifyErrorNo = classifyErrorNo;
//		this.list = list;
//		this.i = i;
//		this.qqs = qqs;
//	}// 构造函数
//
//	public void run() {
//
//		SelectiveEnsemble se = new SelectiveEnsemble();
//		
//		this.list[this.i] = se.CrossValidationModel(input, classifier, numfolds,
//				correctRateArray, random, this.i, this.qqs);
//		for (int j = 0; j < this.list[this.i].size(); j++) {
//			if (this.list[this.i].get(j) == 0) {
//				this.classifyErrorNo[this.i].add(j);
//			}
//		}
//		this.isFinished = true;
//		executeTime = System.currentTimeMillis();
//	}
//
//	public double getValue() {
//		return value;
//	}
//
//	public void setValue(double value) {
//		this.value = value;
//	}
//
//	public List<Integer> getLst() {
//		return lst;
//	}
//
//	public void setLst(List<Integer> lst) {
//		this.lst = lst;
//	}
//
//	public boolean isFinished() {
//		return isFinished;
//	}
//
//	public void setFinished(boolean isFinished) {
//		this.isFinished = isFinished;
//	}
//
//	public int getI() {
//		return i;
//	}
//
//	public void setI(int i) {
//		this.i = i;
//	}
//
//	public long getExecuteTime() {
//		return executeTime;
//	}
//
//	public void setExecuteTime(long executeTime) {
//		this.executeTime = executeTime;
//	}
//	
//	
//}






//class TrainThreads extends Thread {
//	private Instances inputR;
//	private Classifier[] cfsArray;
//	private int numfolds;
//	private List<Double> correctRateArray;
//	private Random random;
//
//	private List<Integer>[] list;
//	private List<Integer>[] classifyErrorNo;
//
//	TrainThreads(Instances inputR, Classifier[] cfsArray, int numfolds,
//			List<Double> correctRateArray, Random random, List<Integer>[] list, List<Integer>[] classifyErrorNo) {
//		this.inputR = inputR;
//		this.cfsArray = cfsArray;
//		this.numfolds = numfolds;
//		this.correctRateArray = correctRateArray;
//		this.random = random;
//		this.list = list;
//		this.classifyErrorNo = classifyErrorNo;
//	}
//
//	public void run() {
////		for (int i = 0; i < cfsArray.length; i++) {
//////			System.out.print("分类器" + i + "	");
////			IndependentTrainThread itt = new IndependentTrainThread(inputR,
////					cfsArray[i], numfolds, correctRateArray, random, list, i, classifyErrorNo);
////			itt.start();
////		}
//	}
//
//	public List<Integer>[] getList() {
//		return list;
//	}
//
//	public void setList(List<Integer>[] list) {
//		this.list = list;
//	}
//
//	public List<Integer>[] getClassifyErrorNo() {
//		return classifyErrorNo;
//	}
//
//	public void setClassifyErrorNo(List<Integer>[] classifyErrorNo) {
//		this.classifyErrorNo = classifyErrorNo;
//	}
//	
//	
//	
//}
