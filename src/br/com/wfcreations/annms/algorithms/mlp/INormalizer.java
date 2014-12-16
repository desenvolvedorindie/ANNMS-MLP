package br.com.wfcreations.annms.algorithms.mlp;

import br.com.wfcreations.sannmf.data.SupervisedSet;

public interface INormalizer {

	public void normalize(SupervisedSet data);

	public double[] normalize(double[] vector);
}
