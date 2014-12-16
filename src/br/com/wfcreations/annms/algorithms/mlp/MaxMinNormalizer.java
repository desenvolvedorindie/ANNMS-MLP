package br.com.wfcreations.annms.algorithms.mlp;

import br.com.wfcreations.sannmf.data.SupervisedPattern;
import br.com.wfcreations.sannmf.data.SupervisedSet;

public class MaxMinNormalizer implements INormalizer {

	double[] max, min;

	public MaxMinNormalizer(SupervisedSet data) {
		int inputSize = data.inputsNum();

		max = new double[inputSize];
		min = new double[inputSize];

		for (int i = 0; i < inputSize; i++) {
			max[i] = Double.MIN_VALUE;
			min[i] = Double.MAX_VALUE;
		}

		for (SupervisedPattern dataSetRow : data.getPatterns()) {
			double[] input = dataSetRow.getInputs();
			for (int i = 0; i < inputSize; i++) {
				if (input[i] > max[i])
					max[i] = input[i];
				if (input[i] < min[i])
					min[i] = input[i];
			}
		}
	}

	@Override
	public void normalize(SupervisedSet data) {
		for (SupervisedPattern row : data.getPatterns()) {
			double[] normalizedInput = normalize(row.getInputs());
			row.setInputs(normalizedInput);
		}
	}

	@Override
	public double[] normalize(double[] vector) {
		double[] normalizedVector = new double[vector.length];
		for (int i = 0; i < vector.length; i++)
			normalizedVector[i] = (vector[i] - min[i]) / (max[i] - min[i]);
		return normalizedVector;
	}
}