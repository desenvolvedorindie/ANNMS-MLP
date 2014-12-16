package br.com.wfcreations.annms.algorithms.mlp;

import br.com.wfcreations.sannmf.data.SupervisedSet;

public interface ISampling {
	public SupervisedSet[] sample(SupervisedSet dataSet);
}
