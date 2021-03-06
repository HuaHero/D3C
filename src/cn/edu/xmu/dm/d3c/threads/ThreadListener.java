package cn.edu.xmu.dm.d3c.threads;

/**
 * desc:轮询各个分类器线程是否结束
 * <code>ThreadListener</code>
 * @author chenwq (chenwq@stu.xmu.edu.cn)
 * @version 1.0 2012/04/10
 */
import java.util.ArrayList;
import java.util.Calendar;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import cn.edu.xmu.dm.d3c.utils.InitClassifiers;

public class ThreadListener extends Thread {
	public static boolean isOver = false;
	public ArrayList<IndependentTrainThread> array = new ArrayList<IndependentTrainThread>();
	long sleepTime = 1000;

	public void run() {
		Logger logger = Logger.getLogger(ThreadListener.class);
		PropertyConfigurator.configure("log4j.properties");
		
		boolean flag = false;
		long startTime = System.currentTimeMillis();
		while (!flag) {
			boolean isRemoved = false;
//			System.err.println("当前线程个数: " + array.size());
			try {// 每隔一定时间监听一次各个文件统计线程
				Thread.sleep(sleepTime);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			for (int i = 0; i < array.size(); i++) {
				if (array.get(i).isFinished()) {// 判断统计文件的线程是否已经完成

//					System.out.println(InitClassifiers.classifiersName[array
//							.get(i).getI()] + " is finished!");
					Calendar c = Calendar.getInstance();
					c.setTimeInMillis(array.get(i).getExecuteTime() - startTime);
//					System.out.println(InitClassifiers.classifiersName[array
//							.get(i).getI()]
//							+ " runs "
//							+ c.get(Calendar.SECOND)
//							+ "s");

					logger.info(InitClassifiers.classifiersName[array.get(i)
							.getI()] + " is finished!");
					logger.info(InitClassifiers.classifiersName[array.get(i)
							.getI()] + " runs " + c.get(Calendar.SECOND) + "s\n");

					array.remove(i);// 将已经完成的线程对象从队列中移除
					isRemoved = true;
				}
				if (!isRemoved) {
					long t2 = System.currentTimeMillis();
					Calendar c = Calendar.getInstance();
					c.setTimeInMillis(t2 - startTime);
					if (c.get(Calendar.SECOND) >= 100) {
						array.get(i).stop();
						array.remove(i);// 将已经完成的线程对象从队列中移除

//						System.out
//								.println(InitClassifiers.classifiersName[array
//										.get(i).getI()] + " is removed!");
						logger.info(InitClassifiers.classifiersName[array
								.get(i).getI()] + " is removed!");
					}
				}
			}
			if (array.size() == 0) {// 如果统计线程都已经完成
				flag = true;
				ThreadListener.isOver = true;
			}
		}
	}
}
