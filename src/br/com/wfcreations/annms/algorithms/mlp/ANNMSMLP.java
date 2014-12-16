package br.com.wfcreations.annms.algorithms.mlp;

import java.util.ArrayList;

import br.com.wfcreations.annms.api.data.*;
import br.com.wfcreations.annms.api.data.type.ListType;
import br.com.wfcreations.annms.api.data.value.*;
import br.com.wfcreations.annms.api.data.value.validate.*;
import br.com.wfcreations.annms.api.lang.ArrayUtils;
import br.com.wfcreations.annms.api.neuralnetwork.*;
import br.com.wfcreations.sannmf.function.activation.IDerivativeActivationFunction;
import br.com.wfcreations.sannmf.function.activation.Sigmoid;
import br.com.wfcreations.sannmf.function.activation.Tanh;
import br.com.wfcreations.sannmf.learning.algorithms.Backpropagation;
import br.com.wfcreations.sannmf.neuralnetwork.MLP;

@NeuralNetwork("MLP")
public class ANNMSMLP implements INeuralNetwork {

	private static final long serialVersionUID = 1L;

	private static final ValueValidate intValueValidate = new ValueValidate(Int.class);

	private static final ValueValidate booleanValueValidate = new ValueValidate(Bool.class);

	private static final ValueValidate idValueValidate = new ValueValidate(ID.class);

	public static final ID TANH_ID = ID.create("TANH");

	public static final ID SIGMOID_ID = ID.create("SIGMOID");

	public static final ID HIDDENS_ID = ID.create("HIDDENS");

	public static final ID HASBIAS_ID = ID.create("HASBIAS");

	public static final ID ACTIVATIONFUNCTION_ID = ID.create("ACTIVATIONFUNCTION");

	public static enum ActivationFunctionType {
		SIGMOID, TANH;

		public IDerivativeActivationFunction getActivationFunction() {
			if (this == SIGMOID) {
				return new Sigmoid();
			} else if (this == TANH) {
				return new Tanh();
			}
			return null;
		}
	}

	public static double[] translateInput(Pattern pattern) {
		double[] input = new double[pattern.getValuesNum()];
		for (int i = 0; i < pattern.getValuesNum(); i++)
			if (pattern.getValueAt(i) instanceof Real)
				input[i] = Real.getValueFor(pattern.getValueAt(i));
			else if (pattern.getValueAt(i) instanceof Int)
				input[i] = Int.getValueFor(pattern.getValueAt(i));
		return input;
	}

	public static double[] translateOutput(Pattern pattern, ListType listType) {
		double[] output = new double[listType.getListValuesNum()];
		output[listType.getIndexOf((ID) pattern.getValueAt(0))] = 1;
		return output;
	}

	private int[] hiddens;

	private boolean hasBias = true;

	private ActivationFunctionType activationFunction = ActivationFunctionType.SIGMOID;

	private MLP mlp;

	private Backpropagation backpropagation;

	private ListType outputs;

	private int wrongClassification;

	private int rightClassification;

	private int trainingSetSize;

	private int testSetSize;

	@Override
	public void create(Param[] params) throws Exception {
		int h;
		for (Param param : params) {
			if (param.idEquals(HIDDENS_ID)) { // HIDENS
				if (param.size() > 0 && intValueValidate.isValid(param.getValues())) {
					this.hiddens = new int[param.size()];
					int i = 0;
					for (IParamValue value : param.getValues()) {
						h = Int.getValueFor((IValue) value);
						if (h > 0)
							this.hiddens[i++] = h;
						else
							throw new Exception("Hiddens units quantity must be greater than 0");
					}
				} else
					throw new Exception(String.format("Invalid %s param value", HIDDENS_ID));
			} else if (param.idEquals(HASBIAS_ID)) { // HASBIAS
				if (param.size() == 1 && booleanValueValidate.isValid(param.getValues()))
					this.hasBias = Bool.getValueFor((IValue) param.getValueAt(0));
				else
					throw new Exception(String.format("Invalid %s param value", HASBIAS_ID));
			} else if (param.idEquals(ACTIVATIONFUNCTION_ID)) { // ACTIVATIONFUNCTION
				if (param.size() == 1 && idValueValidate.isValid(param.getValues()) && (param.getValueAt(0).equals(TANH_ID) || param.getValueAt(0).equals(SIGMOID_ID))) {
					if (param.getValueAt(0).equals(ID.create(ActivationFunctionType.SIGMOID.name())))
						this.activationFunction = ActivationFunctionType.SIGMOID;
					else
						this.activationFunction = ActivationFunctionType.TANH;
				} else
					throw new Exception(String.format("Invalid %s param value", ACTIVATIONFUNCTION_ID));
			} else {
				throw new Exception(String.format("Invalid param %s", param.getID()));
			}
		}

		if (this.hiddens.length == 0)
			throw new Exception("Hiddens unit not defined");
	}

	@Override
	public IValue[] run(IValue[] values) throws Exception {
		Bool trained = this.mlp == null ? Bool.FALSE : Bool.TRUE;
		if (!trained.getValue())
			throw new Exception("Not trained");
		else {
			mlp.setInput(translateInput(new Pattern(values)));
			mlp.activate();
			return new IValue[] { outputs.getValuesAt(ArrayUtils.findMax(mlp.getOutput())) };
		}
	}

	@Override
	public Param[] status() {
		Bool trained = this.mlp == null ? Bool.FALSE : Bool.TRUE;
		ArrayList<Param> statusList = new ArrayList<Param>();

		if (trained.getValue()) {
			statusList.add(new Param(ID.create("INPUTS_NUM"), ArrayUtils.createArray(new Int(mlp.getInputsNum()))));
			statusList.add(new Param(ID.create("OUTPUTS_NUM"), ArrayUtils.createArray(new Int(mlp.getOutputsNum()))));
			int hiddensNum = mlp.getLayers().size() - 2;
			statusList.add(new Param(ID.create("HIDDENS"), ArrayUtils.createArray(new Int(hiddensNum))));
			IParamValue[] hiddensUnits = new IParamValue[hiddensNum];
			for (int i = 1; i < mlp.getLayers().size() - 1; i++)
				hiddensUnits[i - 1] = new Int(mlp.getLayerAt(i).getNeuronsNum());
			statusList.add(new Param(ID.create("HIDDENS_UNITS"), hiddensUnits));
			statusList.add(new Param(ID.create("LEARN_RATE"), ArrayUtils.createArray(new Real(backpropagation.getLearningRate()))));
			statusList.add(new Param(ID.create("CURRENT_EPOCH"), ArrayUtils.createArray(new Int(backpropagation.getCurrentEpoch()))));
			statusList.add(new Param(ID.create("ERROR_FUNCTION"), ArrayUtils.createArray(new Str(backpropagation.getErrorFunction().getClass().toString().toUpperCase()))));
			statusList.add(new Param(ID.create("TOTAL_NETWORK_ERROR"), ArrayUtils.createArray(new Real(backpropagation.getTotalNetworkError()))));
			statusList.add(new Param(ID.create("TRAINING_SET_SIZE"), ArrayUtils.createArray(new Int(this.trainingSetSize))));
			statusList.add(new Param(ID.create("TEST_SET_SIZE"), ArrayUtils.createArray(new Int(this.testSetSize))));
			statusList.add(new Param(ID.create("RIGHT_CLASSIFICATION"), ArrayUtils.createArray(new Int(this.rightClassification))));
			statusList.add(new Param(ID.create("WRONG_CLASSIFICATION"), ArrayUtils.createArray(new Int(this.wrongClassification))));
		}

		statusList.add(new Param(ID.create("TRAINDED"), ArrayUtils.createArray(trained)));

		return statusList.toArray(new Param[statusList.size()]);
	}

	public int[] getHiddens() {
		return hiddens;
	}

	public boolean hasBias() {
		return hasBias;
	}

	public IDerivativeActivationFunction getActivationFunction() {
		return activationFunction.getActivationFunction();
	}

	public MLP getMLP() {
		return mlp;
	}

	public void setMLP(MLP mlp) {
		this.mlp = mlp;
	}

	public Backpropagation getBackpropagation() {
		return backpropagation;
	}

	public void setBackpropagation(Backpropagation backpropagation) {
		this.backpropagation = backpropagation;
	}

	public ListType getOutputs() {
		return outputs;
	}

	public void setOutputs(ListType outputs) {
		this.outputs = outputs;
	}

	public int getWrongClassification() {
		return wrongClassification;
	}

	public void setWrongClassification(int wrongClassification) {
		this.wrongClassification = wrongClassification;
	}

	public int getRightClassification() {
		return rightClassification;
	}

	public void setRightClassification(int rightClassification) {
		this.rightClassification = rightClassification;
	}

	public int getTrainingSetSize() {
		return trainingSetSize;
	}

	public void setTrainingSetSize(int trainingSetSize) {
		this.trainingSetSize = trainingSetSize;
	}

	public int getTestSetSize() {
		return testSetSize;
	}

	public void setTestSetSize(int testSetSize) {
		this.testSetSize = testSetSize;
	}
}