function Yhat = elm_predict(model, X)
    %si applica la centratura del dataset se presente
    if isfield(model, 'mu')
        X = (X - model.mu)./ model.sigma;
    end

	H = model.activation(model.W1 * X' + model.b1);
	Yhat = (model.W2' * H)';
end