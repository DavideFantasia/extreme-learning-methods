% Pulizia dell'ambiente
clear; clc; close all;

% Aggiunta sottocartelle al path
addpath(genpath('algorithms'));
addpath(genpath('utils'));

% Fake dataset
X = rand(100, 5);    % 100 samples, 5 features
Y = sin(X(:,1)) + X(:,2).^2;
    
% Hyperparameters
hidden_dim = 100;
lambda = 1e-3;
    
activation = @(z) tanh(z);
    
% Train
model = elm_train(X, Y, hidden_dim, lambda, activation);
    
% Predict
Yhat = elm_predict(model, X);

%differenza fra valore predetto e valore reale
disp(norm(Y-Yhat));