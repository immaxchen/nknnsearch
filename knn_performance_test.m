
kVal = 20;
nCores = 2;

%% small sample, full run

disp(' ');
disp('small data (10k)');
disp(' ');

N = 10000;
x_train = randn(N,25);

disp('Testing for Distance Sorting');
tic;
parfor t = 1:nCores
    for i = 1:size(x_train,1)
        dist = sum(abs(x_train-repmat(x_train(i,:),N,1)),2);
        [sdist, idx] = sort(dist);
    end
end
toc;

disp('Testing for Linear Search (MatLab version)');
tic;
parfor t = 1:nCores
    for i = 1:size(x_train,1)
        [idx, sdist] = knnLinearSearch(x_train, x_train(i,:), kVal);
    end
end
toc;

disp('Testing for MatLab knnsearch');
tic;
parfor t = 1:nCores
    [idx, sdist] = knnsearch(x_train,x_train,'Distance','cityblock','K',kVal);
end
toc;

disp('Testing for Linear Search (C version)');
tic;
parfor t = 1:nCores
    [I, D] = nknnsearch(x_train.', x_train.', kVal);
    [I, D] = sortnknn(I, D);
end
toc;

%% median sample, half run

disp(' ');
disp('median data (50k)');
disp(' ');

N = 50000;
x_train = randn(N,25);

disp('Testing for MatLab knnsearch');
tic;
parfor t = 1:nCores
    [idx, sdist] = knnsearch(x_train(1:N/2,:),x_train,'Distance','cityblock','K',kVal);
end
toc;

disp('Testing for Linear Search (C version)');
tic;
parfor t = 1:nCores
    [I, D] = nknnsearch(x_train(1:N/2,:).', x_train.', kVal);
    [I, D] = sortnknn(I, D);
end
toc;

%% median sample, half run

disp(' ');
disp('median data (100k)');
disp(' ');

N = 100000;
x_train = randn(N,25);

disp('Testing for MatLab knnsearch');
tic;
parfor t = 1:nCores
    [idx, sdist] = knnsearch(x_train(1:N/2,:),x_train,'Distance','cityblock','K',kVal);
end
toc;

disp('Testing for Linear Search (C version)');
tic;
parfor t = 1:nCores
    [I, D] = nknnsearch(x_train(1:N/2,:).', x_train.', kVal);
    [I, D] = sortnknn(I, D);
end
toc;

%% large sample, partial run

disp(' ');
disp('large data (1M)');
disp(' ');

N = 1000000;
x_train = randn(N,25);

disp('Testing for MatLab knnsearch');
tic;
parfor t = 1:nCores
    [idx, sdist] = knnsearch(x_train,x_train(1:2000,:),'Distance','cityblock','K',kVal);
end
toc;

disp('Testing for Linear Search (C version)');
tic;
parfor t = 1:nCores
    [I, D] = nknnsearch(x_train.', x_train(1:2000,:).', kVal);
    [I, D] = sortnknn(I, D);
end
toc;
