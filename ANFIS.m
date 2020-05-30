% hold on
tic;
clc;
clear;
close all;

%% Load Data
% load dataset
loaddata = csvread('routput.csv',1,0);

Inputs = loaddata(:, 1:end-1);
Targets = loaddata(:, end);

data.Inputs=Inputs;
data.Targets=Targets;

%% Generate Basic FIS

fis=CreateInitialFIS(data,3);

output=evalfis(data.Inputs,fis);
time=toc;
PlotResult(data.Targets,output,'ANFIS',time);