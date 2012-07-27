package cn.edu.xmu.dm.d3c.sample;

import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import cn.edu.xmu.dm.d3c.core.MyUtil;
import cn.edu.xmu.dm.d3c.core.SelectiveStrategy;
import cn.edu.xmu.dm.d3c.threads.ThreadListener;
import cn.edu.xmu.dm.d3c.utils.InitClassifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * desc:D3C集成分类器使用模板
 * <code>Sample</code>
 * @author chenwq (chenwq@stu.xmu.edu.cn)
 * @version 1.0 2012/04/10
 */
public class Sample {

	public static void main(String[] args) throws Exception {

		Logger logger = Logger.getLogger(ThreadListener.class);
		PropertyConfigurator.configure("log4j.properties");

//		String filename = "D:\\data\\feature35_train_1000_1.arff";
		String filename = "data/3762no_2302yes.arff";
//		String filename = "data/german.arff";
//		String filename = "data/bupa.arff";

		logger.info("----------------------- " + filename);
		Date now = new Date();
		SimpleDateFormat f = new SimpleDateFormat("yyyy-MM-dd kk:mm");
		logger.info("----------------------- " + f.format(now));

		// 初始化工具类
		MyUtil myutil = new MyUtil();
		//
		Instances input = myutil.getInstances(filename);
		//
		input.setClassIndex(input.numAttributes() - 1);

		Classifier[] cfsArray = InitClassifiers.init("config/classifiers.xml");
		
		// 定义交叉验证次数
		int numfolds = 5;
		//
		SelectiveStrategy test = new SelectiveStrategy();
		
		/**
		 *  其他可选算法类型:
		 *  1、HCNRR
		 *  2、HCRR
		 *  3、EBSS
		 *  4、EFSS
		 */
		
		test.setSelectiveAlgorithm("CC");

		test.ClusterBasedStrategy(input, cfsArray, numfolds);
	}
}
