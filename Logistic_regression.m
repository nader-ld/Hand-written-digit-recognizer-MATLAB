%% Hand-written-digit-recognition Logistic-Regression-Classifier

% 
%  This file contains code that utilizes logistic regression to recognize
%  hand-written digits. 
% 
%

%% Initialization
clear ; close all; clc

%% Setup the parameters

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                   
%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);


%% ============ Part 2: Training the Logistic Regression ============
%  We will implement one-vs-all classification for the handwritten
%  digit dataset.
%

lambda = 0.1; %regularization
[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

