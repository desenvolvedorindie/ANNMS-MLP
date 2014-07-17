package br.com.wfcreations.annms.algorithms.mlp;

import br.com.wfcreations.annms.api.data.*;
import br.com.wfcreations.annms.api.data.value.*;
import br.com.wfcreations.annms.api.data.value.validate.*;
import br.com.wfcreations.annms.api.neuralnetwork.*;
import br.com.wfcreations.sannmf.function.activation.IDerivativeActivationFunction;
import br.com.wfcreations.sannmf.function.activation.Sigmoid;
import br.com.wfcreations.sannmf.function.activation.Tanh;

@NeuralNetwork("MLP")
public class ANNMSMLP implements INeuralNetwork {

	private static final long serialVersionUID = 1L;

	private static final ValidateAbstract hiddensNeuronsValidate = new IntValidate(true);

	private static final ValidateAbstract hasBiasValidate = new BooleanValidate(false);

	private static final ValidateAbstract activationFunctionValidate = new IDValidate(false);

	private static final ValidateAbstract activationFunctionTypeValidate = new InArrayValidate(new String[] { "TAHN", "SIGMOID" }, true);

	public static final String TAHN = "TAHN";

	public static final String SIGMOID = "SIGMOID";

	public static final String HIDDENS_ID = "HIDDENS";

	public static final String HASBIAS_ID = "HASBIAS";

	public static final String ACTIVATIONFUNCTION_ID = "ACTIVATIONFUNCTION";

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

	private int hiddens[];

	private boolean hasBias = false;

	private ActivationFunction activationFunction = ActivationFunction.TANH;

	@Override
	public void create(Param[] params) throws Exception {
		for (Param param : params) {
			System.out.println(param.getID());
			if (param.idIs(ID.create(HIDDENS_ID)) && param.size() > 0 && hiddensNeuronsValidate.isValid(param.getValues()[0])) {
				this.hiddens = new int[param.getValues().length];
				int i = 0;
				for (IParamValue value : param.getValues())
					if (value instanceof Int)
						this.hiddens[i++] = Int.getValueFor((IValue) value);
					else
						throw new Exception("Invalid param valeu for Hiddens");
			} else if (param.idIs(ID.create(HASBIAS_ID))) {
				if (param.size() == 1 && hasBiasValidate.isValid(param.getValues()[0]))
					this.hasBias = Bool.getValueFor((IValue) param.getValues()[0]);
				else {
					throw new Exception(String.format("Invalid %s param value", HASBIAS_ID));
				}
			} else if (param.idIs(ID.create(ACTIVATIONFUNCTION_ID))) {
				if (param.size() == 1 && activationFunctionValidate.isValid(param.getValues()[0]) && activationFunctionTypeValidate.isValid(param.getValues()[0])) {
					if (ID.getValueFor((IValue) param.getValueAt(0)).equals(ActivationFunction.TANH.name())) {
						this.activationFunction = ActivationFunction.TANH;
					} else {
						this.activationFunction = ActivationFunction.SIGMOID;
					}
				}
			} else {
				throw new Exception(String.format("Invalid param %s", param.getID()));
			}
		}
	}

	public int[] getHiddens() {
		return hiddens;
	}

	public void setHiddens(int hiddens[]) {
		this.hiddens = hiddens;
	}

	public boolean isHasBias() {
		return hasBias;
	}

	public void setHasBias(boolean hasBias) {
		this.hasBias = hasBias;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public void setActivationFunction(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	@Override
	public IValue[] run(IValue[] values) throws Exception {
		throw new IllegalArgumentException("Not trained");
	}

	@Override
	public Param[] status() {
		return new Param[] { new Param(ID.create("TRAINDED"), new IParamValue[] { Bool.FALSE }) };
	}
}