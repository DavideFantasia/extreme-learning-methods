function Yhat = elm_predict(model, X)
	H = model.activation(model.W1 * X' + model.b1);
	Yhat = (model.W2' * H)';
end