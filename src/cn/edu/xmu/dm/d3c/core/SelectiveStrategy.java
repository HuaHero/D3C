package cn.edu.xmu.dm.d3c.core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import cn.edu.xmu.dm.d3c.threads.ClassifierIndex;
import cn.edu.xmu.dm.d3c.threads.ClassifierIndexComparator;
import cn.edu.xmu.dm.d3c.utils.InitClassifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;


public class SelectiveStrategy {
	
	//
	private double ssCorrectRate=1;
	private double ssInterval=0.05;
	private int numClusters=10;
	
	String selectiveAlgorithm="EFSS";
	String CCAlgorithm="EFSS";
	//
	public void setSelectiveAlgorithm(String sa){
		//
		selectiveAlgorithm=sa;
	}
	//
	public void setCircleCombineAlgorithm(String cca){
		//
		CCAlgorithm=cca;
	}
	//
	public void setInitCorrectRate(double correctrate){
		//
		ssCorrectRate=correctrate;
	}
	//
	public void setInitInterval(double interval){
		//
		ssInterval=interval;
	}
	//
	public void setNumClusters(int num){
		//
		numClusters=num;
	}
	
	//
	/*public void ensembleVote(Instances train, Classifier[] newCfsArray) {

		double correctRate =0;
		
		try {
			int i;
			
			Vote ensemble = new Vote();
			//
			SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,
					Vote.TAGS_RULES);
			//
			ensemble.setCombinationRule(tag);
			//
			ensemble.setClassifiers(newCfsArray);
			//
			ensemble.setSeed(2);
			//
			ensemble.buildClassifier(train);
			//
			Evaluation eval = new Evaluation(train);
			//
			Random random = new Random();
			//
			eval.crossValidateModel(ensemble, train, 5, random);
			//
			System.out.println(1-eval.errorRate());
			System.out.println(eval.toMatrixString());
			
			

		} catch (Exception e) {
			e.printStackTrace();
		}
	}*/
	
	
	//简单决策
	public void SimpleStrategy(Instances input, Classifier[] cfsArray, int numfolds)throws Exception{
		int i, j;
		// 用于记录每种分类器单独训练时分类情况，1表示分类正确，0表示分类错误
		List<Integer>[] list = new List[cfsArray.length];
		// 用于得到每种分类器单独训练时候错分实例的编号
		List<Integer>[] classifyErrorNo = new List[cfsArray.length];
		// 用来存放各种算法的正确率
		List<Double> correctRateArray = new ArrayList();
		//
		SelectiveEnsemble se = new SelectiveEnsemble();
		//
		List<ClassifierIndex> qqs = new ArrayList<ClassifierIndex>();
		se.IndependentTrain(input, cfsArray, numfolds, list, classifyErrorNo,correctRateArray, qqs);
		//
		double currentCorrectRate=0;
		double initCorrectRate = ssCorrectRate;
		double interval = ssInterval;
		//
		List<Integer> ClassifierNo = new ArrayList();//
		//
		List<Double> currentResult = new ArrayList();//
		currentResult.add(Double.MAX_VALUE);
		currentResult.add((double) 0);
		currentResult.add((double) 0);
		//
		SeletiveAlgorithm sa = new SeletiveAlgorithm();
		//
		if(selectiveAlgorithm.equals("HCNRR"))
			currentCorrectRate=sa.HCNRR(input,cfsArray,list,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("HCRR"))
			currentCorrectRate=sa.HCNRR(input,cfsArray,list,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("EBSS"))
			currentCorrectRate = sa.EBSS(input, cfsArray, list, correctRateArray,initCorrectRate, currentResult, ClassifierNo);
		else if(selectiveAlgorithm.equals("EFSS"))
			currentCorrectRate = sa.EFSS(input, cfsArray, list,correctRateArray,initCorrectRate, currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("CC"))
			currentCorrectRate=sa.CircleCombine(input,cfsArray,list,correctRateArray,initCorrectRate,interval,currentResult,ClassifierNo,CCAlgorithm);
		else throw new Exception ("Could not find selective algorithm:"+selectiveAlgorithm);
		
		//
		DecimalFormat df = new DecimalFormat("0.00000");
		System.out.println(df.format(currentCorrectRate));
		System.out.println(ClassifierNo);
	}
	
	//基于聚类的选择策略
	public void ClusterBasedStrategy(Instances input, Classifier[] cfsArray, int numfolds)throws Exception{
		Logger logger = Logger.getLogger(SelectiveStrategy.class);
		PropertyConfigurator.configure("log4j.properties");
		
		int i, j;
		// 用于记录每种分类器单独训练时分类情况，1表示分类正确，0表示分类错误
		List<Integer>[] list = new List[cfsArray.length];
		// 用于得到每种分类器单独训练时候错分实例的编号
		List<Integer>[] classifyErrorNo = new List[cfsArray.length];
		// 用来存放各种算法的正确率
		List<Double> correctRateArray = new ArrayList();
	
		List<ClassifierIndex> qqs = new ArrayList<ClassifierIndex>();
		
		//
		SelectiveEnsemble se = new SelectiveEnsemble();
		//
		se.IndependentTrain(input, cfsArray, numfolds, list, classifyErrorNo,correctRateArray, qqs);
		Collections.sort(qqs, new ClassifierIndexComparator());  
		for (Iterator iterator = qqs.iterator(); iterator.hasNext();) {
			ClassifierIndex qq = (ClassifierIndex) iterator.next();
//			System.out.println(InitClassifiers.classifiersName[qq.getIndex()] + "'s correct rate:\t\t\t" + qq.getCorrectRate());
			
			logger.info(InitClassifiers.classifiersName[qq.getIndex()] + "'s correct rate:\t\t\t" + qq.getCorrectRate());
		}
	
		
		
		
		correctRateArray = new ArrayList();
		for(int ii = 0; ii < qqs.size(); ii++){
			correctRateArray.add(qqs.get(ii).getCorrectRate());
		}
//		System.out.println("分类器正确率:");
//		for (Iterator iterator = correctRateArray.iterator(); iterator.hasNext();) {
//			Double qq = (Double) iterator.next();
//			System.out.print(qq + "\t");
//			
//		}
		// 初始化工具类
		MyUtil myutil = new MyUtil();
		//
		// 产生分类结果的文件
		myutil.createClassifyResultFile(input.numInstances(), list);
		// 将分类结果读入形成表示各个分类器的实例
		Instances classifyResult = myutil.getInstances("ClassifyResult.arff");
		//
		MyKMeans km = new MyKMeans();
		List<Integer> chooseClassifiers = new ArrayList();
		km.setNumClusters(numClusters);
		km.buildClusterer(classifyResult, chooseClassifiers, correctRateArray);
		//
//		System.out.println();
//		System.out.println("聚类结果：");
		
		logger.info("\n classifier results:");
		
//		System.out.print("[");
		logger.info("[");
		for(i = 0; i < chooseClassifiers.size(); i++){
//			System.out.print(InitClassifiers.classifiersName[chooseClassifiers.get(i)] + ",");
			logger.info(InitClassifiers.classifiersName[chooseClassifiers.get(i)] + ",");
		}
//		System.out.println("]");
		logger.info("]\n");
		// 用来得到通过聚类后选中的分类器的分类结果
		List<Integer>[] newList = new List[chooseClassifiers.size()];
		for (i = 0; i < chooseClassifiers.size(); i++) {
			newList[i] = list[chooseClassifiers.get(i)];
			// System.out.println(newList[i]);
		}
		// 用来得到通过聚类后选中的分类器的错分实例编号
		List<Integer>[] newClassifyErrorNo = new List[chooseClassifiers.size()];
		for (i = 0; i < chooseClassifiers.size(); i++) {
			newClassifyErrorNo[i] = classifyErrorNo[chooseClassifiers.get(i)];
		}
		// 用来存放各种算法的正确率
		List<Double> newCorrectRateArray = new ArrayList();
		for (i = 0; i < chooseClassifiers.size(); i++) {
			newCorrectRateArray.add(correctRateArray.get(chooseClassifiers.get(i)));
		}
		//
		Classifier[] newCfsArray = new Classifier[chooseClassifiers.size()];
		for (i = 0; i < chooseClassifiers.size(); i++) {
			newCfsArray[i] = cfsArray[chooseClassifiers.get(i)];
		}
		//
		double newInitCorrectRate = ssCorrectRate;
		double newInterval = ssInterval;
		//
		List<Integer> ClassifierNo = new ArrayList();//
		//
		List<Double> currentResult = new ArrayList();//
		currentResult.add(Double.MAX_VALUE);
		currentResult.add((double) 0);
		currentResult.add((double) 0);
		//
		double currentCorrectRate = 0;
		//
		SeletiveAlgorithm sa = new SeletiveAlgorithm();
		//
		if(selectiveAlgorithm.equals("HCNRR"))
			currentCorrectRate=sa.HCNRR(input, newCfsArray, newList,newCorrectRateArray,newInitCorrectRate, currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("HCRR"))
			currentCorrectRate=sa.HCRR(input, newCfsArray, newList,newCorrectRateArray,newInitCorrectRate, currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("EBSS"))
			currentCorrectRate=sa.EBSS(input, newCfsArray, newList,newCorrectRateArray,newInitCorrectRate, currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("EFSS"))
			currentCorrectRate=sa.EFSS(input, newCfsArray, newList,newCorrectRateArray,newInitCorrectRate, currentResult,ClassifierNo);
		else if(selectiveAlgorithm.equals("CC"))
			currentCorrectRate = sa.CircleCombine(input, newCfsArray,newList, newCorrectRateArray, newInitCorrectRate,newInterval, currentResult, ClassifierNo,CCAlgorithm);
		else throw new Exception ("Could not find selective algorithm:"+selectiveAlgorithm);
		
		//
		//
		System.out.println();
		System.out.println();
		//
		DecimalFormat df = new DecimalFormat("0.00000");
		System.out.println("精确度："+df.format(currentCorrectRate));
		System.out.println("分类器组合");
		System.out.print("[");
		for(i = 0; i < ClassifierNo.size(); i++){
			System.out.print(InitClassifiers.classifiersName[ClassifierNo.get(i)] + ",");
		}
		System.out.println("]");
//		System.out.println(sa.bestMatrixString);
		logger.info(sa.bestMatrixString);
//		System.out.println(sa.bestClassDetailsString);
		logger.info(sa.bestClassDetailsString);
	}
}

