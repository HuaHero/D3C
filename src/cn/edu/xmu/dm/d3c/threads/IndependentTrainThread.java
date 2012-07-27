package cn.edu.xmu.dm.d3c.threads;
import java.util.List;
import java.util.Random;

import cn.edu.xmu.dm.d3c.core.SelectiveEnsemble;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * desc:各个分类器单独训练模型
 * <code>IndependentTrainThread</code>
 * @author chenwq (chenwq@stu.xmu.edu.cn)
 * @version 1.0 2012/04/10
 */
public class IndependentTrainThread extends Thread {
	private double value;
	private Instances input;
	private Classifier classifier;
	private int numfolds;
	private List<Double> correctRateArray;
	private Random random;
	private List<Integer> lst;
	private List<Integer>[] classifyErrorNo;
	private List<Integer>[] list;
	private int i;
	private List<ClassifierIndex> qqs;
	
	private long executeTime;
	
	private boolean isFinished;
	
	public IndependentTrainThread(Instances input, Classifier classifier,
			int numfolds, List<Double> correctRateArray, Random random, List<Integer>[] list, int i, List<Integer>[] classifyErrorNo, List<ClassifierIndex> qqs) {
		this.input = input;
		this.classifier = classifier;
		this.numfolds = numfolds;
		this.correctRateArray = correctRateArray;
		this.random = random;
		this.classifyErrorNo = classifyErrorNo;
		this.list = list;
		this.i = i;
		this.qqs = qqs;
	}// 构造函数

	public void run() {

		SelectiveEnsemble se = new SelectiveEnsemble();
		
		this.list[this.i] = se.CrossValidationModel(input, classifier, numfolds,
				correctRateArray, random, this.i, this.qqs);
		for (int j = 0; j < this.list[this.i].size(); j++) {
			if (this.list[this.i].get(j) == 0) {
				this.classifyErrorNo[this.i].add(j);
			}
		}
		this.isFinished = true;
		executeTime = System.currentTimeMillis();
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

	public List<Integer> getLst() {
		return lst;
	}

	public void setLst(List<Integer> lst) {
		this.lst = lst;
	}

	public boolean isFinished() {
		return isFinished;
	}

	public void setFinished(boolean isFinished) {
		this.isFinished = isFinished;
	}

	public int getI() {
		return i;
	}

	public void setI(int i) {
		this.i = i;
	}

	public long getExecuteTime() {
		return executeTime;
	}

	public void setExecuteTime(long executeTime) {
		this.executeTime = executeTime;
	}
}
