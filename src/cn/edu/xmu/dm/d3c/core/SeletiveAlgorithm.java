package cn.edu.xmu.dm.d3c.core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.collections.CollectionUtils;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import cn.edu.xmu.dm.d3c.utils.InitClassifiers;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SeletiveAlgorithm {
	
	//第一个分类器是否为最优
	private boolean bestBegin=true;
	//
	public String tempMatrixString="";
	//
	public String bestMatrixString="";
	//
	public String tempClassDetailsString="";
	//
	public String bestClassDetailsString="";
	
	//设置bestBegin
	public void setBestBegin(boolean bb){
		bestBegin=bb;
	}
	//
	public void setTempMatrixString(String str){
		//
		tempMatrixString=str;
	}
	//
	public void setBestMatrixString(String str){
		//
		bestMatrixString=str;
	}
	//
	public void setTempClassDetailsString(String str){
		//
		tempClassDetailsString=str;
	}
	//
	public void setBestClassDetailsString(String str){
		//
		bestClassDetailsString=str;
	}
	
	
	//
	public double ensembleVote(Instances train, /*Instances test,*/Classifier[] newCfsArray) {

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
			
			/*Instance testInst;
			Evaluation eval = new Evaluation(train);
			eval.setPriors(train);
			//
			for (i = 0; i < test.numInstances(); i++) {
				//
				testInst = test.instance(i);
				//
				eval.evaluateModelOnceAndRecordPrediction(ensemble, testInst);
			}*/
			correctRate = 1 - eval.errorRate();
			setTempMatrixString(eval.toMatrixString());
			setTempClassDetailsString(eval.toClassDetailsString());
			

		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return correctRate;
	}
	
	/* 爬山策略（非重复随机）
	 * 参数意义和爬山策略（重复随机）相同，爬山策略（重复随机）和爬山策略（不重复随机）主要的不同在于抽取分类器的方式上，有点像不重复采样和重复采样
	 */
	public double HCNRR(Instances train,Classifier[] cfsArray,List<Integer>[] classifyResult,List<Double> correctRateArray,double initCorrectRate,List<Double> currentResult,List<Integer> ClassifierNo){
		//
		double diversity;
		double tempDiversity;
		double correctRate;
		double voteCorrectRate;
		//
		int i=0,j,k;
		int r;
		int maxNo;
		int tempNo;
		int candidateNo;
		List<Integer> candidateClassifierNo=new ArrayList();//候选分类器列表初始化基本包括全部的分类器编号，其中分类器编号会在比较之后删除
		Random random=new Random();
		//如果列表为空，说明是第一轮循环，那么就决定是分类效果最好的分类器还是最近产生的分类器为第一个分类器，并将除了第一个分类器以外的所有分类器当作候选分类器，
		//否则就只要将正确率替换成前一轮循环得到的正确率，并将全部的分类器当作候选分类器
		if (ClassifierNo.size() == 0) {
			
			diversity=currentResult.get(0);
			//
			if (bestBegin == false) {
				r = random.nextInt(cfsArray.length);
				correctRate = correctRateArray.get(r);
				ClassifierNo.add(r);
				tempNo=r;
			} else {
				correctRate = Collections.max(correctRateArray);
				maxNo = correctRateArray.indexOf(correctRate);
				ClassifierNo.add(maxNo);
				tempNo=maxNo;
			}
			for(j=0;j<cfsArray.length;j++){
				if (j != tempNo) {
					candidateClassifierNo.add(j);
				}	
			}
		} else {
			diversity=currentResult.get(0);
			correctRate = currentResult.get(1);
			//
			for(j=0;j<cfsArray.length;j++){
				candidateClassifierNo.add(j);
			}
		}
		//
		while (candidateClassifierNo.size() != 0) {
			if (correctRate >= initCorrectRate) {
				break;
			} else {
				// 以候选分类器的数量作为参数,随机产生一个与之前没有重复的分类器编号
				r = random.nextInt(candidateClassifierNo.size());
				candidateNo = candidateClassifierNo.get(r);
				// 将得到该分类器和之前选择的分类器的分类结果一起计算差异性
				List<Integer>[] tempList = new List[ClassifierNo.size() + 1];
				for (k = 0; k < ClassifierNo.size(); k++) {
					tempList[k] = classifyResult[ClassifierNo.get(k)];
				}
				tempList[ClassifierNo.size()] = classifyResult[candidateNo];
				tempDiversity = CalculateK(tempList);
				//
				if (tempDiversity <= diversity) {
					//
					ClassifierNo.add(candidateNo);
					//
					Classifier[] newCfsArray = new Classifier[ClassifierNo.size()];
					//
					for (j = 0; j < newCfsArray.length; j++) {
						newCfsArray[j] = cfsArray[ClassifierNo.get(j)];
					}
					//
					voteCorrectRate = ensembleVote(train, /* test, */newCfsArray);
					//
					if (voteCorrectRate > correctRate) {
						diversity = tempDiversity;
						correctRate = voteCorrectRate;
						setBestMatrixString(tempMatrixString);
						setBestClassDetailsString(tempClassDetailsString);
					} else {
						ClassifierNo.remove(ClassifierNo.size() - 1);
					}
				}
				//
				candidateClassifierNo.remove(r);
			}
		}
		//说明只有一个分类器，差异性为0
		if(diversity==Double.MAX_VALUE){
			diversity=0;
		}
		//
		currentResult.clear();
		currentResult.add(diversity);
		currentResult.add(correctRate);
		currentResult.add((double)ClassifierNo.size());
		
		return correctRate;
	}
	
	/* 爬山策略（重复随机）
	 * train为训练集
	 * cfsArray为全部候选分类器
	 * classifyResult为每个分类器对训练集单独训练得到的结果（0表示错误，1表示正确），用于计算差异度
	 * correctRateArray为每个分类器单独训练时候得到的正确率
	 * initCorrectRate为初始的目标正确率，在循环集成中随着每次循环不断递减（对循环集成起效）
	 * currentResult为当前一轮循环得到的结果(currentRersult.get[0]表示差异性度量值，currentResult.get[1]表示正确率，currentResult.get[2]为得到的分类器的数量，对循环集成起效)
	 * ClassifierNo为当前一轮循环得到的分类器的编号序列
	 */
	public double HCRR(Instances train,Classifier[] cfsArray,List<Integer>[] classifyResult,List<Double> correctRateArray,double initCorrectRate,List<Double> currentResult,List<Integer> ClassifierNo){
		double diversity;//差异性（差异性度量是κ度量，κ值越小，差异性越大）
		double tempDiversity;
		double correctRate;//正确率
		double voteCorrectRate;//vote集成得到的正确率
		//
		int count=cfsArray.length;//循环次数(循环次数为分类器的个数)
		int i=0,j,k;
		int r;
		int maxNo;
		Random random=new Random();
		//如果列表为空，说明是第一轮循环，那么就将正确率最高的分类器选为第一个分类器,否则就只要将正确率替换成前一轮循环得到的正确率
		if (ClassifierNo.size() == 0) {
			//
			diversity=currentResult.get(0);
			//
			if (bestBegin == false) {
				r = random.nextInt(cfsArray.length);
				correctRate = correctRateArray.get(r);
				ClassifierNo.add(r);
			} else {
				correctRate = Collections.max(correctRateArray);
				maxNo = correctRateArray.indexOf(correctRate);
				ClassifierNo.add(maxNo);
			}
		} else {
			diversity=currentResult.get(0);
			correctRate=currentResult.get(1);
		}
		//
		while (i < count) {
			//
			if (correctRate >= initCorrectRate) {
				break;
			} else {
				// 随机产生一个分类器的编号，然后将得到该分类器和之前选择的分类器的分类结果一起计算差异性
				r = random.nextInt(cfsArray.length);
				List<Integer>[] tempList = new List[ClassifierNo.size() + 1];
				for (k = 0; k < ClassifierNo.size(); k++) {
					tempList[k] = classifyResult[ClassifierNo.get(k)];
				}
				tempList[ClassifierNo.size()] = classifyResult[r];
				tempDiversity = CalculateK(tempList);
				// κ度量的值越小，说明差异性越大
				if (tempDiversity <= diversity) {
					//
					ClassifierNo.add(r);
					//
					Classifier[] newCfsArray = new Classifier[ClassifierNo.size()];
					//
					for (j = 0; j < newCfsArray.length; j++) {
						newCfsArray[j] = cfsArray[ClassifierNo.get(j)];
					}
					//
					voteCorrectRate = ensembleVote(train, /* test, */newCfsArray);
					// 如果当前集成得到的正确率比之前的正确率高，那么就将差异性和正确率替换成该轮的值，否则将最后一个分类器编号删除(新加的分类器编号)
					if (voteCorrectRate > correctRate) {
						diversity = tempDiversity;
						correctRate = voteCorrectRate;
						setBestMatrixString(tempMatrixString);
						setBestClassDetailsString(tempClassDetailsString);
					} else {
						ClassifierNo.remove(ClassifierNo.size() - 1);
					}
				}
				i++;
			}
		}
		// 说明只有一个分类器，差异性为0
		if (diversity == Double.MAX_VALUE) {
			diversity = 0;
		}
		//对于循环集成，每一次循环结束必须记录当前的差异性、正确率和所选分类器个数，对于所选择的分类编号训练由ClassifierNo保存
		currentResult.clear();
		currentResult.add(diversity);
		currentResult.add(correctRate);
		currentResult.add((double)ClassifierNo.size());
		
		return correctRate;
	}
	
	//集成后序选择
	public double EBSS(Instances train,Classifier[] cfsArray,List<Integer>[] classifyResult,List<Double> correctRateArray,double initCorrectRate,List<Double> currentResult,List<Integer> ClassifierNo){
		//
		double diversity;
		double tempDiversity=Double.MAX_VALUE;
		double correctRate;
		double voteCorrectRate;
		List<Double> tempResult=new ArrayList();
		//
		int i=0,j,k,r;
		int index;
		int threshold=2*cfsArray.length;
		double num;
		/*
		 * 第一阶段
		 * 如果已选的分类器的个数为0，说明是第一次进行后序选择，那么将全部的分类作为候选分类器加入列表中，并计算此时的差异性和正确率，
		 * 否则说明已经经过至少1次的循环，那么首先将差异性和正确率替换成前一次的结果，然后判断此时是否满足正确率阈值。
		 * 如果不满足，则将全部分类器一个一个的进入列表，同时记录目前为止最大的差异性和对应的分类器的数量（分类器数量是不超过阈值的）。
		 * 如果分类器数量超过阈值，那么列表中的分类器减少到之前差异性最大时候的分类器数量。
		 * 之前的操作完成之后，候选分类器已经确定，那么计算此时的正确率，进入第二阶段。
		 */
		if(ClassifierNo.size()==0){
			diversity=CalculateK(classifyResult);
			correctRate=ensembleVote(train, /*test,*/cfsArray);
			//
			currentResult.clear();
			currentResult.add(diversity);
			currentResult.add(correctRate);
			currentResult.add((double)cfsArray.length);
			//
			for(j=0;j<cfsArray.length;j++){
				ClassifierNo.add(j);
			}
		}else{
			//
			diversity=currentResult.get(0);
			correctRate=currentResult.get(1);
			//
			if (correctRate >= initCorrectRate) {
				return correctRate;
			}
			//tempResult.get[0]得到当前最高的差异性，tempResult.get[1]得到对应的分类器数量
			tempResult.add(diversity);
			//tempResult.add(correctRate);
			tempResult.add(currentResult.get(2));
			//
			for(j=0;j<cfsArray.length;j++){
				//
				List<Integer>[] tempList = new List[ClassifierNo.size() + 1];
				//
				for(k=0;k<ClassifierNo.size();k++){
					tempList[k]=classifyResult[ClassifierNo.get(k)];
				}
				tempList[ClassifierNo.size()]=classifyResult[j];
				//
				tempDiversity = CalculateK(tempList);
				//
				ClassifierNo.add(j);
				//差异度量κ越小，说明差异性越大，进行集成的效果可能更好
				if((tempDiversity<tempResult.get(0)) && (ClassifierNo.size()<=threshold)){
					//
					tempResult.clear();
					tempResult.add(tempDiversity);
					tempResult.add((double)ClassifierNo.size());
				}
			}
			diversity=tempDiversity;
			//分类器数量超过阈值
			if(ClassifierNo.size()>threshold){
				//
				while(ClassifierNo.size()!=tempResult.get(1)){
					//
					ClassifierNo.remove(ClassifierNo.size()-1);
				}
				diversity=tempResult.get(0);
			}
			//
			Classifier[] newCfsArray = new Classifier[ClassifierNo.size()];
			for (k = 0; k < ClassifierNo.size(); k++) {
				newCfsArray[k] = cfsArray[ClassifierNo.get(k)];
			}
			correctRate = ensembleVote(train,/* test, */newCfsArray);
		}
		setBestMatrixString(tempMatrixString);
		setBestClassDetailsString(tempClassDetailsString);
		/*
		 * 第二阶段
		 * 本程序的后序选择的删除操作，是根据分类器的类型（0~19）进行。
		 * 首先查找对应类型分类器（从0开始）在列表中的位置，如果返回值是-1，说明给类型的分类器在列表中不存在，那么就查找下一种分类器类型。
		 * 如果返回的不为-1，那么就将对应位置上的分类器屏蔽，然后计算正确率，如果此时的正确率高于之前的正确率，那么就将该位置上的分类器从
		 * 列表中删除，并将正确率替换成最新计算得到的正确率。
		 */
		r=0;
		//
		while (r != cfsArray.length) {
			if (correctRate >= initCorrectRate) {
				break;
			} else {
				//System.out.println("round:" + r);
				//
				index = ClassifierNo.indexOf(r);
				//
				if (index != -1) {
					//
					j=0;
					List<Integer>[] tempList = new List[ClassifierNo.size()-1];
					for(k=0;k<ClassifierNo.size();k++){
						if(k!=index){
							tempList[j]=classifyResult[ClassifierNo.get(k)];
							j++;
						}
					}
					//
					tempDiversity = CalculateK(tempList);
					//
					if (tempDiversity <= diversity) {
						Classifier[] newCfsArray = new Classifier[ClassifierNo.size() - 1];
						j = 0;
						for (k = 0; k < ClassifierNo.size(); k++) {
							if (k != index) {
								newCfsArray[j] = cfsArray[ClassifierNo.get(k)];
								j++;
							}
						}
						//
						voteCorrectRate = ensembleVote(train,/* test, */newCfsArray);
						//
						if (voteCorrectRate >= correctRate) {
							//
							ClassifierNo.remove(index);
							diversity=tempDiversity;
							correctRate = voteCorrectRate;
							setBestMatrixString(tempMatrixString);
							setBestClassDetailsString(tempClassDetailsString);
							
						}
					}
				}
				//
				r++;
			}
		}
		// 说明只有一个分类器，差异性为0
		if (diversity == Double.MAX_VALUE) {
			diversity = 0;
		}
		//
		currentResult.clear();
		currentResult.add(diversity);
		currentResult.add(correctRate);
		currentResult.add((double)ClassifierNo.size());
		//
		System.out.println(correctRate);
		//System.out.println(bestMatrixString);
		
		return correctRate;
	}
	
	/* 集成前序选择
	 * 参数意义同爬山策略（重复随机）
	 * 集成前序选择是将正确率比较好的分类器选出，与之前获得的分类器一起计算差异性和正确率，然后判断分类器是否被选中
	 */
	public double EFSS(Instances train,Classifier[] cfsArray,List<Integer>[] classifyResult,List<Double> correctRateArray,double initCorrectRate,List<Double> currentResult,List<Integer> ClassifierNo){
		//
		double diversity;//差异性
		double tempDiversity;
		double correctRate;//正确率
		double voteCorrectRate=0;//vote集成得到的正确率
		//
		List<Integer> sortedNo=new ArrayList();//按正确率排序之后的分类器编号
		int i=0,j,k;
		int tempNo,currentNo;
		int threshold=2*cfsArray.length;
		//对正确率进行降序，并且获得排序之后每个位置上正确率的对应分类器编号
		/****************************/
		List<Double> newCorrectRateArray=new ArrayList();
		List<Double> tempCorrectRateArray=new ArrayList();
		newCorrectRateArray.addAll(correctRateArray);
		tempCorrectRateArray.addAll(correctRateArray);
		List<Integer> temp=new ArrayList();
		for(i=0;i<tempCorrectRateArray.size();i++){
			temp.add(i);
		}
		Collections.sort(newCorrectRateArray);
		for(i=newCorrectRateArray.size()-1;i>=0;i--){
			tempNo=tempCorrectRateArray.indexOf(newCorrectRateArray.get(i));
			sortedNo.add( temp.get(tempNo));
			tempCorrectRateArray.remove(tempNo);
			temp.remove(tempNo);
		}
		/****************************/
		//System.out.println(sortedNo);
		//
		if (ClassifierNo.size() == 0) {
			diversity=currentResult.get(0);
			ClassifierNo.add(sortedNo.get(0));
			correctRate = correctRateArray.get(sortedNo.get(0));
			sortedNo.remove(0);
		} else {
			diversity=currentResult.get(0);
			correctRate=currentResult.get(1);
		}
		//
		while (sortedNo.size() != 0) {
			
			if (correctRate >= initCorrectRate) {
				break;
			} else {
				// 获得当前正确率最高的分类器的编号，将它与之前选中的分类器一起计算差异性
				currentNo = sortedNo.get(0);
				List<Integer>[] tempList = new List[ClassifierNo.size() + 1];
				for (k = 0; k < ClassifierNo.size(); k++) {
					tempList[k] = classifyResult[ClassifierNo.get(k)];
				}
				tempList[ClassifierNo.size()] = classifyResult[currentNo];
				tempDiversity = CalculateK(tempList);
				//
				if (tempDiversity <= diversity) {
					//
					ClassifierNo.add(sortedNo.get(0));
					Classifier[] newCfsArray = new Classifier[ClassifierNo.size()];
					for (j = 0; j < newCfsArray.length; j++) {
						newCfsArray[j] = cfsArray[ClassifierNo.get(j)];
					}
					voteCorrectRate = ensembleVote(train, /* test, */newCfsArray);
					//
					if (voteCorrectRate > correctRate) {
						diversity = tempDiversity;
						correctRate = voteCorrectRate;
						//
						setBestMatrixString(tempMatrixString);
						setBestClassDetailsString(tempClassDetailsString);
					} else {
						ClassifierNo.remove(ClassifierNo.size() - 1);
					}
				}
				sortedNo.remove(0);
			}
		}
		// 说明只有一个分类器，差异性为0
		if (diversity == Double.MAX_VALUE) {
			diversity = 0;
		}
		//
		currentResult.clear();
		currentResult.add(diversity);
		currentResult.add(correctRate);
		currentResult.add((double)ClassifierNo.size());
		//
		System.out.println(correctRate);
		//System.out.println(bestMatrixString);
		
		return correctRate;
	}
	
	//原动态选择
	public double DS(Instances train,Classifier[] cfsArray,List<Integer> D,List<Double> correctRateArray,double initCorrectRate,List<Double> currentResult,List<Integer> ClassifierNo){
		double correctRate;
		int k;
		int threshold=2*cfsArray.length;
		//
		List<Integer> tempD=new ArrayList();
		tempD.addAll(D);
		//
		ClassifierNo.add(tempD.get(0));
		correctRate=correctRateArray.get(tempD.get(0));
		tempD.remove(0);
		//
		//
		while (tempD.size() != 0) {
			//
			if(ClassifierNo.size()>threshold){
				correctRate=currentResult.get(1);
				while(ClassifierNo.size()!=currentResult.get(2)){
					ClassifierNo.remove(ClassifierNo.size()-1);
				}
			}
			//
			if (correctRate >= initCorrectRate) {
				break;
			} else {
				
				ClassifierNo.add(tempD.get(0));
				
				Classifier[] newCfsArray = new Classifier[ClassifierNo.size()];
				
				for(k=0;k<ClassifierNo.size();k++){
					newCfsArray[k]=cfsArray[ClassifierNo.get(k)];
				}
				
				correctRate = ensembleVote(train, /* test, */newCfsArray);
					
				tempD.remove(0);
				
				if((correctRate>currentResult.get(1)) && (ClassifierNo.size()<=threshold)){
					currentResult.clear();
					currentResult.add(Double.MAX_VALUE);
					currentResult.add(correctRate);
					currentResult.add((double)ClassifierNo.size());
				}
			}
		}
		//
		currentResult.clear();
		currentResult.add((double)0);
		currentResult.add(correctRate);
		currentResult.add((double)ClassifierNo.size());
		//
		System.out.println();
		DecimalFormat df=new DecimalFormat("0.00000");
		//System.out.println(df.format(correctRate));
		
		return correctRate;
		
	}
	
	//循环集成框架
	public double CircleCombine(Instances train,Classifier[] cfsArray,List<Integer>[] classifyResult,List<Double> correctRateArray,double initCorrectRate, double interval,List<Double> currentResult,List<Integer> ClassifierNo,String CCAlgorithm)throws Exception{
		Logger logger = Logger.getLogger(SeletiveAlgorithm.class);
		PropertyConfigurator.configure("log4j.properties");
		
		//List<Integer> ClassifierNo=new ArrayList();//用来存放选到的分类器
		//List<Double> currentResult=new ArrayList();//用来存放最近一次循环得到的结果,currentResult.get[0]表示差异性,currentResult.get[1]表示正确率,currentResult.get[2]表示当前的分类器个数
		List<Integer> OptimalNo=new ArrayList();
		List<Double> optimalResult=new ArrayList();//用来存放全部最优的结果,optimalResult.get[0]表示差异性,optimalResult.get[1]表示正确率
		//
		//currentResult.add(Double.MAX_VALUE);
		//currentResult.add((double)0);
		//currentResult.add((double)0);
		//
		optimalResult.add(Double.MAX_VALUE);
		optimalResult.add((double)0);
		//
		int circle=0;
		int i;
		int position;
		double maxCorrectRate;
		
		//
		DecimalFormat df = new DecimalFormat("0.00000");
		//
		while(initCorrectRate>=0){
			//
			System.out.println();
			System.out.println();
			//
//			System.out.println("Circle:"+circle+"	当前最优精度:"+df.format(optimalResult.get(1))+"	当前目标精度:"+df.format(initCorrectRate));
			logger.info("Circle:"+circle+"	best precision now:"+df.format(optimalResult.get(1))+"	target precision:"+df.format(initCorrectRate));
			//
			if(CCAlgorithm.equals("HCNRR"))
				HCNRR(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			else if(CCAlgorithm.equals("HCRR"))
				HCRR(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			else if(CCAlgorithm.equals("EBSS"))
				EBSS(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			else if(CCAlgorithm.equals("EFSS"))
				EFSS(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			else throw new Exception ("Could not find selective algorithm:"+CCAlgorithm);
			
			
			//HCNRR(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			//HCRR(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			//EBSS(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			//EFSS(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			//DS(train,cfsArray,classifyResult,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			//
			if(currentResult.get(1)>optimalResult.get(1)){
				optimalResult.clear();
				optimalResult.add(currentResult.get(0));
				optimalResult.add(currentResult.get(1));
				OptimalNo.clear();
				OptimalNo.addAll(ClassifierNo);
			}
			//
			String strClassifierNo = "[";
			if(optimalResult.get(1)>=initCorrectRate){
				//System.out.println();
				//
				//System.out.print(initCorrectRate+"	");
				//System.out.print(currentResult.get(1)+"	");
//				System.out.println(ClassifierNo);
			
				for(int ii = 0; ii < ClassifierNo.size(); ii++){
					strClassifierNo += InitClassifiers.classifiersName[ClassifierNo.get(ii)] + ", ";
				}
				//
				/*System.out.print(initCorrectRate+"	");
				System.out.print(optimalResult.get(1)+"	");
				System.out.println(OptimalNo);*/
				break;
			}else{
				//System.out.println();
				//
				//System.out.print(initCorrectRate+"	");
				//System.out.print(currentResult.get(1)+"	");
//				System.out.println(ClassifierNo);

				for(int ii = 0; ii < ClassifierNo.size(); ii++){
					strClassifierNo += InitClassifiers.classifiersName[ClassifierNo.get(ii)] + ", ";
				}
				//
				/*System.out.print(initCorrectRate+"	");
				System.out.print(optimalResult.get(1)+"	");
				System.out.println(OptimalNo);*/
				initCorrectRate=initCorrectRate-interval;	
			}
			/*if(currentResult.get(1)>=initCorrectRate){
				System.out.println("initCorrectRate:"+initCorrectRate);
				System.out.println("currentCorrectRate:"+currentResult.get(1));
				System.out.println("ClassifierNo:"+ClassifierNo);
				break;
			}else{
				System.out.println("initCorrectRate:"+initCorrectRate);
				System.out.println("currentCorrectRate:"+currentResult.get(1));
				System.out.println("ClassifierNo:"+ClassifierNo);
				initCorrectRate=initCorrectRate-interval;
			}*/
			//
			circle++;
			strClassifierNo += "]";
			logger.info(strClassifierNo);
		}
		
		
		System.out.println();
		//System.out.println("所得精度："+df.format(optimalResult.get(1)));
		//System.out.println("对应分类器组合："+OptimalNo);
		//System.out.println(bestMatrixString);
		
		
		return optimalResult.get(1);
		
	} 
	
	//特别的循环集成框架
	public double CC(Instances train,Classifier[] cfsArray,List<Integer> D,List<Double> correctRateArray,double initCorrectRate, double interval,List<Double> currentResult,List<Integer> ClassifierNo){
		//List<Integer> ClassifierNo=new ArrayList();//用来存放选到的分类器
		//List<Double> currentResult=new ArrayList();//
		//	
		int circle=0;
		//
		double correctRate=0;
		//
		while(initCorrectRate>=0){
			//
			System.out.println("Circle:"+circle);
			//
			correctRate=DS(train,cfsArray,D,correctRateArray,initCorrectRate,currentResult,ClassifierNo);
			//
			if(correctRate>=initCorrectRate){
				System.out.print(initCorrectRate+"	");
				System.out.print(correctRate+"	");
				System.out.println("ClassifierNo:"+ClassifierNo);
				break;
			}else{
				System.out.print(initCorrectRate+"	");
				System.out.print(correctRate+"	");
				System.out.println("ClassifierNo:"+ClassifierNo);
				initCorrectRate=initCorrectRate-interval;
			}
			//
			circle++;
		}
		return correctRate;
		
	}
	
	//计算两个分类器之间的不一致度量
	public double CalculateDis(List<Integer> first,List<Integer> second){
		double Dis=0;
		int i;
		int diffNum=0;
		//
		for(i=0;i<first.size();i++){
//			//
//			if(first.get(i)!=second.get(i)){
//				diffNum=diffNum+1;
//			}
			if(second.size() > i){
				if(first.get(i)!=second.get(i)){
					diffNum=diffNum+1;
				}
			}else{
				break;
			}
		}
		//
		Dis= (double)diffNum/(double)first.size();
		//System.out.println(Dis);
		return Dis;
	}
	
	//计算多个分类器之间的κ度量
	public double CalculateK(List<Integer>[] classifyResult){
		
		int L=classifyResult.length;
		int N=classifyResult[0].size();
		int i,j;
		int num=0;
		//
		double Dis=0;
		double p;
		double k;
		//计算DISav
		for(i=0;i<L-1;i++){
			//
			for(j=i+1;j<L;j++){
				//System.out.print("("+i+","+j+")"+"	");
				Dis=Dis+CalculateDis(classifyResult[i],classifyResult[j]);
			}
			//System.out.println();
		}
		Dis=(Dis*2)/(double)(L*(L-1));
		//计算平均准确率
		for(i=0;i<classifyResult.length;i++){
			//
			for(j=0;j<classifyResult[i].size();j++){
				//
				if(classifyResult[i].get(j)==1){
					num=num+1;
				}
			}
		}
		p=(double)num/(double)(L*N);
		
		//System.out.println("Dis:"+Dis);
		//System.out.println("p:"+p);
		
		
		//计算κ度量
		k=1-Dis/(2*p*(1-p));
		//System.out.println(k);
		return k;
	}	
	
}



