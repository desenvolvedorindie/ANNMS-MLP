package br.com.wfcreations.annms.algorithms.mlp;

import br.com.wfcreations.annms.api.data.*;
import br.com.wfcreations.annms.api.neuralnetwork.*;

@LearningRule("BACKPROPAGATION")
public class ANNMSBackpropagation implements ISupervisedLearningRule {

	private static final long serialVersionUID = 1L;

	@Override
	public void create(Param[] params) throws Exception {
	}

	@Override
	public INeuralNetwork train(INeuralNetwork network, Data inputs, Data outputs) {
		return network;
	}
}