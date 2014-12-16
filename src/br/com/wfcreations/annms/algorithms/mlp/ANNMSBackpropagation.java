package br.com.wfcreations.annms.algorithms.mlp;

import java.util.ArrayList;
import java.util.List;

import br.com.wfcreations.annms.api.data.Data;
import br.com.wfcreations.annms.api.data.Param;
import br.com.wfcreations.annms.api.data.type.ListType;
import br.com.wfcreations.annms.api.data.value.Bool;
import br.com.wfcreations.annms.api.data.value.ID;
import br.com.wfcreations.annms.api.data.value.IValue;
import br.com.wfcreations.annms.api.data.value.Int;
import br.com.wfcreations.annms.api.data.value.Real;
import br.com.wfcreations.annms.api.data.value.validate.ValueValidate;
import br.com.wfcreations.annms.api.lang.ArrayUtils;
import br.com.wfcreations.annms.api.neuralnetwork.INeuralNetwork;
import br.com.wfcreations.annms.api.neuralnetwork.ISupervisedLearningRule;
import br.com.wfcreations.annms.api.neuralnetwork.LearningRule;
import br.com.wfcreations.sannmf.data.SupervisedPattern;
import br.com.wfcreations.sannmf.data.SupervisedSet;
import br.com.wfcreations.sannmf.function.weightinitialization.UniformDistribution;
import br.com.wfcreations.sannmf.learning.algorithms.Backpropagation;
import br.com.wfcreations.sannmf.learning.stopcondition.IStopCondition;
import br.com.wfcreations.sannmf.learning.stopcondition.MaximumEpoch;
import br.com.wfcreations.sannmf.neuralnetwork.MLP;

@LearningRule("BACKPROPAGATION")
public class ANNMSBackpropagation implements ISupervisedLearningRule {

	private static final long serialVersionUID = 1L;

	private static final ValueValidate intValueValidate = new ValueValidate(Int.class);

	private static final ValueValidate realValueValidate = new ValueValidate(Real.class);

	private static final ValueValidate booleanValueValidate = new ValueValidate(Bool.class);

	public static final ID EPOCH_ID = ID.create("EPOCH");

	public static final ID LEARNRATE_ID = ID.create("LEARNRATE");

	public static final ID BATCHMODE_ID = ID.create("BATCHMODE");

	private int epoch;

	private double learnRate;

	private boolean bacthMode = true;

	@Override
	public void create(Param[] params) throws Exception {
		for (Param param : params) {
			if (param.idEquals(EPOCH_ID)) { // EPOCH
				if (param.size() == 1 && intValueValidate.isValid(param.getValues()))
					this.epoch = Int.getValueFor((IValue) param.getValues()[0]);
				else
					throw new Exception(String.format("Invalid %s param value", EPOCH_ID));
			} else if (param.idEquals(LEARNRATE_ID)) { // LEARNRATE
				if (param.size() == 1 && realValueValidate.isValid(param.getValues()))
					this.learnRate = Real.getValueFor((IValue) param.getValueAt(0));
				else
					throw new Exception(String.format("Invalid %s param value", LEARNRATE_ID));
			} else if (param.idEquals(BATCHMODE_ID)) { // BATCHMODE
				if (param.size() == 1 && booleanValueValidate.isValid(param.getValues())) {
					this.bacthMode = Bool.getValueFor((IValue) param.getValueAt(0));
				} else
					throw new Exception(String.format("Invalid %s param value", BATCHMODE_ID));
			} else {
				throw new Exception(String.format("Invalid param %s", param.getID()));
			}
		}
	}

	@Override
	public INeuralNetwork train(INeuralNetwork network, Data inputs, Data outputs) {
		if (!(network instanceof ANNMSMLP))
			throw new IllegalArgumentException("Backpropagation only accept MLP Model");

		ANNMSMLP annmsmlp = (ANNMSMLP) network;

		if (!validateDataInput(inputs))
			throw new IllegalArgumentException("Invalid inputs");
		if (!validateDataOuput(outputs))
			throw new IllegalArgumentException("Invalid outputs");

		SupervisedSet set = getPatterns(inputs, outputs);

		int inputsNum = set.inputsNum();
		int outputsNum = set.outputsNum();

		DataSampling dataSampling = new DataSampling(70);
		SupervisedSet[] sets = dataSampling.sample(set);

		MLP mlp = new MLP(inputsNum, annmsmlp.getHiddens(), outputsNum, annmsmlp.hasBias(), annmsmlp.getActivationFunction());
		mlp.initializeWeights(new UniformDistribution(-1, 1));
		
		Backpropagation backpropagation = new Backpropagation(mlp, learnRate, bacthMode);
		List<IStopCondition> stopConditions = new ArrayList<IStopCondition>();
		stopConditions.add(new MaximumEpoch(backpropagation, epoch));
		INormalizer normalizer = new MaxMinNormalizer(sets[0]);
		normalizer.normalize(sets[0]);
		backpropagation.learn(sets[0], stopConditions);
		
		int right = 0;
		int wrong = 0;
		
		for(int i = 0; i < sets[1].lenght(); i++) {
			mlp.setInput(normalizer.normalize(sets[1].getPatternAt(i).getInputs()));
			mlp.activate();
			if(ArrayUtils.findMax(mlp.getOutput()) == ArrayUtils.findMax(sets[1].getPatternAt(i).getOutputs()))
				right++;
			else
				wrong++;
		}
		annmsmlp.setMLP(mlp);
		annmsmlp.setBackpropagation(backpropagation);
		annmsmlp.setOutputs((ListType) outputs.getAttributeAt(0).getType());
		annmsmlp.setTrainingSetSize(sets[0].lenght());
		annmsmlp.setTestSetSize(sets[1].lenght());
		annmsmlp.setRightClassification(right);
		annmsmlp.setWrongClassification(wrong);
		return annmsmlp;
	}

	protected boolean validateDataInput(Data inputs) {
		if(inputs.getAttributesNum() < 1)
			return false;
		/*
		for (int i = 0; i < inputs.getAttributesNum(); i++) {
			if (!inputs.getAttributeAt(i).getType() == Primitive.INT  || !inputs.getAttributeAt(i).getType().equals(Primitive.REAL))
				return false;
		}
		*/
		return true;
	}

	protected boolean validateDataOuput(Data outputs) {
		if (outputs.getAttributesNum() != 1 || !(outputs.getAttributeAt(0).getType() instanceof ListType))
			return false;
		return true;
	}

	protected SupervisedSet getPatterns(Data inputs, Data outputs) {
		SupervisedSet supervisedSet = new SupervisedSet(inputs.getAttributesNum(), ((ListType) outputs.getAttributeAt(0).getType()).getListValuesNum());
		for (int i = 0; i < inputs.getPatternsNum(); i++)
			supervisedSet.addPattern(new SupervisedPattern(ANNMSMLP.translateInput(inputs.getPatternAt(i)), ANNMSMLP.translateOutput(outputs.getPatternAt(i), (ListType) outputs.getAttributeAt(0).getType())));
		return supervisedSet;
	}
}