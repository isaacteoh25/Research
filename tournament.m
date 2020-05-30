
function [parent1,parent2] = tournament(BestMSE,children,nPop,nChange)

s = randperm(nPop,nChange);%randsample(nPop,nChange);
parents = children(s,:,:,:);
[~,ind] = min(reshape(BestMSE(s),[2,2]));
parent1 = parents(ind(1),:,:,:);
parent2 = parents(ind(2)+2,:,:,:);
end
