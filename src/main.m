% Pulizia dell'ambiente
clear; clc; close all;

% Aggiunta sottocartelle al path
addpath(genpath('algorithms'));
addpath(genpath('utils'));

% algoritmo scelto per la risoluzione del sistema lineare: 'cg' o 'qr'
method = 'qr';

% Fake dataset
X = rand(1000, 2);    % 1000 samples, 2 features
Y = sin(X(:,1)) + X(:,2).^2;
    
% Hyperparameters
hidden_dim = 200;
lambda = 1e-3;
    
activation = @(z) tanh(z);
    
% Train
model = elm_train(X, Y, hidden_dim, lambda, activation, method);
    
% Predict
Yhat = elm_predict(model, X);

%differenza fra valore predetto e valore reale
disp(norm(Y-Yhat));