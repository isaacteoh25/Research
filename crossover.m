function [child1,child2] = crossover(parent1,parent2,class)

    for i = 1:class
        for j = 1:length(parent1)
    alpha = rand();
    child1(:,:,i,j) = alpha*(parent1(:,:,i,j))+(1-alpha)*(parent2(:,:,i,j));
    child2(:,:,i,j) = alpha*(parent2(:,:,i,j))+(1-alpha)*(parent1(:,:,i,j));
        end
    end

end 