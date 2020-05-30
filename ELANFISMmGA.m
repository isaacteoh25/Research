
tic;
clear;
clc;
close all;

% load dataset
data = csvread('routput.csv',1,0);
input = data(:, 1:end-1);
target = data(:, end);

% Initialize parameter

[~, features] = size(input);
class = 3; % [can be change]
ep = 1;
epMax = 400; % [can be change]

nPop = 5;

parents = zeros(nPop, 3, class, features);

pop(1:nPop,:,:,:)= getPopulation(class,input,nPop);
%mGA

count = 0; % generations count with same fitness

output=0;
m0=5;% generations num in which mean remains constant
BestMSE = repmat(100, nPop, 1);
meanLog = 0;
nChange = 4;        % changing population num

vel = zeros(nPop, 3, class, features);
c1 = 1.2;
c2 = 1.2;
while ep < epMax && count < epMax/2
    
    
    % calculate fitness function
for i=1:nPop
    popPosition = squeeze(pop(i, :, :, :));
    [MSE,output] = getFitness(popPosition, class, input,target);
    if MSE <BestMSE(i)
       BestMSE(i) = MSE;
       PBestpos(i, :, :, :) = popPosition;
       Pbestoutput(:,i)=output;
    end
end

    %mGA

    % find gBest
    [bestSol, idx] = min(BestMSE);
    GBestpos = squeeze(PBestpos(idx, :, :, :));
    bestParent = pop(idx,:,:,:); % Define best individual
    bestoutput=Pbestoutput(:,idx);
    
    meanV = sum(BestMSE)/nPop;    % Mean of the cost function
        %----------------------   CONVERGENCE CHECK   -----------------------------
    if length(meanLog) > m0
        curmean = sum(meanLog(ep-m0+1:ep))/m0;
    else
        curmean = sum(meanLog)/length(meanLog);
    end
    
    if abs(abs(curmean)-abs(meanV)) <= 0.05*abs(meanV) || rem(ep,10) == 0
        % Estimate random mf parameters
        [pop(1:nChange,:,:,:)]= getPopulation(class, input,nChange);

    end
   

    r1 = rand();
    r2 = rand();
    for k = 1:2:nChange

    [parents(k,:,:,:),parents(k+1,:,:,:)] = tournament(BestMSE,pop,nPop,nChange);

    [pop( k, :, :,:),pop( k+1, :, :,:)] = crossover(parents(k,:,:,:),parents(k+1,:,:,:),class);
    % vi(t + 1) = wvi(t) + c1r1(pbi(t) - pi(t)) + c2r2(pg(t) - pi(t))
    % pi(t + 1) = pi(t) + vi(t + 1)
    vel(k, :, :, :) = squeeze(vel(k, :, :, :)) + ((c1 * r1) .* (squeeze(PBestpos(k, :, :, :)) - squeeze(pop(k, :, :, :)))) + ((c2 * r2) .* (GBestpos(:, :, :) - squeeze(pop(k, :, :, :))));
    vel(k+1, :, :, :) = squeeze(vel(k+1, :, :, :)) + ((c1 * r1) .* (squeeze(PBestpos(k+1, :, :, :)) - squeeze(pop(k+1, :, :, :)))) + ((c2 * r2) .* (GBestpos(:, :, :) - squeeze(pop(k+1, :, :, :))));
    pop(k, :, :, :) = pop(k, :, :, :) + vel(k ,:, :, :);
    pop(k+1, :, :, :) = pop(k+1, :, :, :) + vel(k+1 ,:, :, :);

    end


    pop(nPop,:,:,:) = bestParent;

    ep = ep + 1;
    Et(ep) = bestSol; 
    meanLog(ep) = meanV; 
    if abs(Et(ep)- Et(ep-1)) <= 0.001*abs(Et(ep))
        count = count+1;
    else
        count = 0;
    end
    plot(1:ep, Et);
    title(['Epoch  ' int2str(ep) ' -> MSE = ' num2str(Et(ep))]);
    grid
    pause(0.001);
    disp(['Iteration ' num2str(ep) ': Best Cost = ' num2str(Et(ep))]);


end
 figure;

time = toc;

PlotResult(target, bestoutput,'ELANFIS-MmGA', time)




