tic;
clear;
clc;
close all;

% load dataset
data = csvread('routput.csv',1,0);
input = data(:, 1:end-1);
target = data(:, end);

% Parameter initialization
[center,U] = fcm(input, 3, [2 100 1e-5]); %center = center cluster, U = membership level
[~, features] = size(input);
class = 3; % [can be change]
ep = 0;
epMax = 400; % [can be change]

% Population initialization
nPop = 5;
% pop = zeros(nPop, 3, class, features); % parameter: population size * 6 * total classes * total features
vel = zeros(nPop, 3, class, features); % velocity matrix of an iteration

c1 = 1.2;
c2 = 1.2;

pop(1:nPop,:,:,:)= getPopulation(class,input,nPop);
output=0;

%PSO
%inisialise pBest
BestMSE = repmat(100, nPop, 1);
PBestpos = zeros(nPop, 3, class, features);

% calculate fitness function
for i=1:nPop
    popPosition = squeeze(pop(i, :, :, :));
    [MSE,output] = getFitness(popPosition, class, input, target);
    if MSE < BestMSE(i)
        BestMSE(i) = MSE;
        PBestpos(i, :, :, :) = popPosition;
        Pbestoutput(:,i)=output;
    end
end
%PSO

% find gBest
[bestSol, idx] = min(BestMSE);
GBestpos = squeeze(PBestpos(idx, :, :, :));

% ITERATION
while ep < epMax
    ep = ep + 1;
    
    % calculate velocity and update particle
    % vi(t + 1) = wvi(t) + c1r1(pbi(t) - pi(t)) + c2r2(pg(t) - pi(t))
    % pi(t + 1) = pi(t) + vi(t + 1)
    r1 = rand();
    r2 = rand();
    for i=1:nPop
        vel(i, :, :, :) = squeeze(vel(i, :, :, :)) + ((c1 * r1) .* (squeeze(PBestpos(i, :, :, :)) - squeeze(pop(i, :, :, :)))) + ((c2 * r2) .* (GBestpos(:, :, :) - squeeze(pop(i, :, :, :))));
        pop(i, :, :, :) = pop(i, :, :, :) + vel(i ,:, :, :);
    end
    
    % calculate fitness value and update pBest
    for i=1:nPop
        popPosition = squeeze(pop(i, :, :, :));
        [MSE,output] = getFitness(popPosition, class, input, target);
        if MSE < BestMSE(i)
            BestMSE(i) = MSE;
            PBestpos(i, :, :, :) = popPosition;
            Pbestoutput(:,i)=output;
        end
    end
    
    % find gBest
    [bestSol, idx] = min(BestMSE);
    GBestpos = squeeze(PBestpos(idx, :, :, :));
    bestoutput=Pbestoutput(:,idx);
    
    Et(ep) = bestSol;


    % Draw the SSE plot
    plot(1:ep, Et);
    title(['Epoch  ' int2str(ep) ' -> MSE = ' num2str(Et(ep))]);
    grid
    pause(0.001);
    disp(['Iteration ' num2str(ep) ': Best Cost = ' num2str(Et(ep))]);

end
 figure;

time = toc;

PlotResult(target, bestoutput,'ELANFIS-PSO', time)