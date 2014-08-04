package br.com.wfcreations.annms.algorithms.mlp;

import java.util.List;
import java.util.Vector;

import br.com.wfcreations.annms.api.data.*;
import br.com.wfcreations.annms.api.data.value.*;
import br.com.wfcreations.annms.api.data.value.validate.*;
import br.com.wfcreations.annms.api.lang.ArrayUtils;
import br.com.wfcreations.annms.api.neuralnetwork.*;
import br.com.wfcreations.sannmf.function.activation.IDerivativeActivationFunction;
import br.com.wfcreations.sannmf.function.activation.Sigmoid;
import br.com.wfcreations.sannmf.function.activation.Tanh;

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

	public static final ID CONNECTINPUTSTOOUTPUTS_ID = ID.create("CONNECTINPUTSTOOUTPUTS");

	public static enum ActivationFunction {
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

	private List<Integer> hiddens = new Vector<>();

	private boolean hasBias = false;

	private ActivationFunction activationFunction = ActivationFunction.TANH;

	private boolean connectInputsToOutputs = false;

	@Override
	public void create(Param[] params) throws Exception {
		int h;
		for (Param param : params) {
			if (param.idEquals(HIDDENS_ID)) {
				if (param.size() > 0 && intValueValidate.isValid(param.getValues()))
					for (IParamValue value : param.getValues()) {
						h = Int.getValueFor((IValue) value);
						if (h > 0)
							this.hiddens.add(h);
						else
							throw new Exception("Hiddens units quantity must be greater than 0");
					}
				else
					throw new Exception(String.format("Invalid %s param value", HIDDENS_ID));
			} else if (param.idEquals(HASBIAS_ID)) {
				if (param.size() == 1 && booleanValueValidate.isValid(param.getValues()))
					this.hasBias = Bool.getValueFor((IValue) param.getValueAt(0));
				else
					throw new Exception(String.format("Invalid %s param value", HASBIAS_ID));
			} else if (param.idEquals(ACTIVATIONFUNCTION_ID)) {
				if (param.size() == 1 && idValueValidate.isValid(param.getValues()) && (param.getValueAt(0).equals(TANH_ID) || param.getValueAt(0).equals(SIGMOID_ID))) {
					if (param.getValueAt(0).equals(ID.create(ActivationFunction.SIGMOID.name())))
						this.activationFunction = ActivationFunction.SIGMOID;
					else
						this.activationFunction = ActivationFunction.TANH;
				} else
					throw new Exception(String.format("Invalid %s param value", ACTIVATIONFUNCTION_ID));
			} else if (param.idEquals(CONNECTINPUTSTOOUTPUTS_ID)) {
				if (param.size() == 1 && booleanValueValidate.isValid(param.getValues()))
					this.connectInputsToOutputs = Bool.getValueFor((IValue) param.getValueAt(0));
				else
					throw new Exception(String.format("Invalid %s param value", CONNECTINPUTSTOOUTPUTS_ID));
			} else {
				throw new Exception(String.format("Invalid param %s", param.getID()));
			}
		}

		if (this.hiddens.size() == 0)
			throw new Exception("Hiddens unit not defined");
	}

	public List<Integer> getHiddens() {
		return hiddens;
	}

	public boolean isHasBias() {
		return hasBias;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public IValue[] run(IValue[] values) throws Exception {
		throw new IllegalArgumentException("Not trained");
	}

	@Override
	public Param[] status() {
		return ArrayUtils.createArray(new Param(ID.create("TRAINDED"), ArrayUtils.createArray(Bool.FALSE)));
	}

	public boolean isConnectInputsToOutputs() {
		return connectInputsToOutputs;
	}
}