
% ===== config and testing compiler =====

mex -setup
copyfile([matlabroot,'/extern/examples/mex/yprime.c'])
mex yprime.c
which yprime
yprime(1, 1:4)

% ===== compile nknn kernel =====

mex nknnsearch.c
mex -v COPTIMFLAGS="-O3" nknnsearch.c
