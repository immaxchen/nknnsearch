function [I,D] = sortnknn(I,D)

[D, II] = sort(D);
[m, n] = size(I);
I = I(sub2ind([m n],II,repmat(1:n,m,1)));

end
