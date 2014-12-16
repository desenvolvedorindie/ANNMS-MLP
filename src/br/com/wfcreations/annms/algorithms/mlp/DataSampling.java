package br.com.wfcreations.annms.algorithms.mlp;

import java.util.ArrayList;
import java.util.Collections;

import br.com.wfcreations.sannmf.data.SupervisedSet;

public class DataSampling implements ISampling {

	protected int percent;

	public DataSampling(int percent) {
		this.percent = percent;
	}

	@Override
	public SupervisedSet[] sample(SupervisedSet dataSet) {
		SupervisedSet[] subSets = new SupervisedSet[2];
		ArrayList<Integer> randoms = new ArrayList<>();
		for (int i = 0; i < dataSet.lenght(); i++) {
			randoms.add(i);
		}
		Collections.shuffle(randoms);

		int inputSize = dataSet.inputsNum();
		int outputSize = dataSet.outputsNum();
		
		subSets[0] = new SupervisedSet(inputSize, outputSize);
		int trainingElementsCount = dataSet.lenght() * percent / 100;
		for (int i = 0; i < trainingElementsCount; i++) {
			int idx = randoms.get(i);
			subSets[0].addPattern(dataSet.getPatternAt(idx));
		}
		
		subSets[1] = new SupervisedSet(inputSize, outputSize);
		int testElementsCount = dataSet.lenght() - trainingElementsCount;
		for (int i = 0; i < testElementsCount; i++) {
			int idx = randoms.get(trainingElementsCount + i);
			subSets[1].addPattern(dataSet.getPatternAt(idx));
		}
		return subSets;
	}
}