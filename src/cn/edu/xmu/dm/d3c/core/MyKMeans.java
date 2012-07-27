package cn.edu.xmu.dm.d3c.core;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
//import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;


public class MyKMeans extends RandomizableClusterer implements NumberOfClustersRequestable, WeightedInstancesHandler {
	//
	private int m_NumClusters=2;
	//
	private Instances m_ClusterCentroids;	
	//
	protected DistanceFunction m_DistanceFunction=new EuclideanDistance();
	//
	private boolean m_PreserveOrder=true;
	//
	private int m_Iterations=0;
	//
	private int m_MaxIterations=100;
	//
	private boolean m_displayStdDevs;
	//
	private Instances m_ClusterStdDevs;
	//
	private int[] m_ClusterSizes;
	
	
	
	//构造函数
	public MyKMeans(){
		super();
		m_SeedDefault=10;
		setSeed(m_SeedDefault);
	}
	//
	public void buildClusterer(Instances data){	
	}
	//
	public void buildClusterer(Instances data,List<Integer> chooseClassifiers,List<Double> correctRateArray)throws Exception{
		//
		m_Iterations=0;
		//
		Instances instances=new Instances(data);
		//
		m_ClusterCentroids=new Instances(instances,m_NumClusters);
		//
		int[] clusterAssignments=new int[instances.numInstances()];
		//
		m_DistanceFunction.setInstances(instances);
		Random RandomO=new Random(getSeed());
		int instIndex;
		HashMap initC=new HashMap();
		DecisionTableHashKey hk=null;
		Instances initInstances=null;
		//
		if(m_PreserveOrder)
			initInstances=new Instances(instances);
		else
			initInstances=instances;
		
		//随机获得起始的聚类中心
		for(int j=initInstances.numInstances()-1;j>=0;j--){
			instIndex=RandomO.nextInt(j+1);
			hk=new DecisionTableHashKey(initInstances.instance(instIndex),initInstances.numAttributes(),true);
			if(!initC.containsKey(hk)){
				m_ClusterCentroids.add(initInstances.instance(instIndex));
				initC.put(hk, null);
			}
			initInstances.swap(j, instIndex);
			
			if(m_ClusterCentroids.numInstances()==m_NumClusters){
				break;
			}
		}
		//
		initInstances=null;
		int i;
		boolean converged=false;
		int emptyClusterCount;
		int q=0;
		Instances[] tempI=new Instances[m_NumClusters];
		//
		//
		//for(q=0;q<instances.numInstances();q++){
		//	System.out.println(instances.instance(q));
		//}
		while(!converged){
			emptyClusterCount=0;
			m_Iterations++;
			converged=true;
			//
			for(i=0;i<instances.numInstances();i++){
				Instance toCluster=instances.instance(i);
				//
				int newC=clusterProcessedInstance(toCluster/*,true*/);
				if(newC!=clusterAssignments[i]){
					converged=false;
				}
				clusterAssignments[i]=newC;
			}
			//改变聚类中心
			m_ClusterCentroids=new Instances(instances,m_NumClusters);
			
			for(i=0;i<m_NumClusters;i++){
				tempI[i]=new Instances(instances,0);
			}
			for(i=0;i<instances.numInstances();i++){
				tempI[clusterAssignments[i]].add(instances.instance(i));
			}
			for(i=0;i<m_NumClusters;i++){
				if(tempI[i].numInstances()==0){
					emptyClusterCount++;
				}else{
					moveCentroid(i,tempI[i]/*,true*/);
				}
			}
			//
			if(emptyClusterCount>0){
				m_NumClusters-=emptyClusterCount;
				//
				if(converged){
					Instances[] t=new Instances[m_NumClusters];
					int index=0;
					for(int k=0;k<tempI.length;k++){
						if(tempI[k].numInstances()>0){
							t[index++]=tempI[k];
						}
					}
					tempI=t;
				}else{
					tempI=new Instances[m_NumClusters];
				}
			}
			//
			if(m_Iterations==m_MaxIterations)
				converged=true;
		}
		//
		/*for (q = 0; q < instances.numInstances(); q++) {
			System.out.println(instances.instance(q));
		}*/
		//
		m_ClusterSizes=new int[m_NumClusters];
		//
		for(i=0;i<m_NumClusters;i++){
			m_ClusterSizes[i]=tempI[i].numInstances();
		}
		/*for(i=0;i<m_ClusterSizes.length;i++){
			System.out.println(m_ClusterSizes[i]);
		}
		for(i=0;i<clusterAssignments.length;i++){
			System.out.print(clusterAssignments[i]+"	");
		}
		System.out.println();*/
		//
		selectClassifier(clusterAssignments,chooseClassifiers,correctRateArray);
	}
	//通过聚类结果，从每个聚类中获得正确率最高的分类器
	public  void selectClassifier(int[] clusterAssignments,List<Integer> chooseClassifiers,List<Double> correctRateArray){
		int i,j;
		double correctRate;
		int chooseID=0;
		
		for(i=0;i<m_NumClusters;i++){
			correctRate=0;
			for(j=0;j<clusterAssignments.length;j++){
//				System.out.println("-----------------------" + clusterAssignments.length);
				if(clusterAssignments[j]==i){
					if(correctRate<correctRateArray.get(j)){
						correctRate=correctRateArray.get(j);
						chooseID=j;
					}
				}
			}
			chooseClassifiers.add(chooseID);
		}
	}
	//移动聚类中心
	protected double[] moveCentroid(int centroidIndex,Instances members/*,boolean updateClusterInfo*/){
		double[] vals=new double[members.numAttributes()];
		
		for(int j=0;j<members.numAttributes();j++){
			if(m_DistanceFunction instanceof EuclideanDistance || members.attribute(j).isNominal()){
				vals[j]=members.meanOrMode(j);
			}
		}
		//
		m_ClusterCentroids.add(decideCentroid(vals,members));
		return vals;
	} 
	//
	public Instance decideCentroid(double[] vals,Instances members){
		
		Instance inst=new Instance(1.0,vals);
		//Instance inst=new DenseInstance(vals.length);
		int q;
		for(q=0;q<vals.length;q++){
			inst.setValue(q, vals[q]);
		}
		//
		double minDistance=Double.MAX_VALUE;
		double tempDistance;
		int instanceID=0;
		int i;
		for(i=0;i<members.numInstances();i++){
			tempDistance=myDistance(inst,members.instance(i));
			//
			if(tempDistance<minDistance ){
				minDistance=tempDistance;
				instanceID=i;
			}
		}
		return members.instance(instanceID);
	}
	//确定实例属于哪一个聚类
	private int clusterProcessedInstance(Instance instance/*,boolean updateErrors*/){
		
		double minDist=Integer.MAX_VALUE;
		
		int bestCluster=0;
		
		for(int i=0;i<m_NumClusters;i++){
			//
			//double dist=m_DistanceFunction.distance(instance,m_ClusterCentroids.instance(i));
			double dist=myDistance(instance,m_ClusterCentroids.instance(i));
			//
			if(dist<minDist){
				minDist=dist;
				bestCluster=i;
			}
		}
		return bestCluster;
	}
	//根据新距离公式得到每个实例之间的距离
	protected  double myDistance(Instance first,Instance second){
		int i;
		int errorIntersect=0;
		for(i=0;i<first.numAttributes();i++){
			if((first.value(i)==second.value(i))&&(first.value(i)==0)){
				++errorIntersect;
			}
		}
		return 1/(double)errorIntersect;
	}
	//获得聚类个数
	public int numberOfClusters()throws Exception{
		return m_NumClusters;
	}
	//设置聚类数
	public void setNumClusters(int n)throws Exception{
		
		if(n<=0){
			throw new Exception("Number of clusters must be > 0");
		}
		m_NumClusters=n;
	}
}
