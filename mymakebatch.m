x0=dlmread('E:\workspace\MLProject\0.txt',',');
x1=dlmread('E:\workspace\MLProject\1.txt',',');
x2=dlmread('E:\workspace\MLProject\2.txt',',');
x3=dlmread('E:\workspace\MLProject\3.txt',',');
x4=dlmread('E:\workspace\MLProject\4.txt',',');
x5=dlmread('E:\workspace\MLProject\5.txt',',');
x6=dlmread('E:\workspace\MLProject\6.txt',',');
x7=dlmread('E:\workspace\MLProject\7.txt',',');
x8=dlmread('E:\workspace\MLProject\8.txt',',');
x9=dlmread('E:\workspace\MLProject\9.txt',',');

digitdata=[]; 
targets=[]; 

digitdata = [digitdata; x0];targets =  [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(x0,1), 1)];
digitdata = [digitdata; x1]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(x1,1), 1)];
digitdata = [digitdata; x2]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(x2,1), 1)]; 
digitdata = [digitdata; x3]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(x3,1), 1)];
digitdata = [digitdata; x4]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(x4,1), 1)]; 
digitdata = [digitdata; x5]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(x5,1), 1)];
digitdata = [digitdata; x6]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(x6,1), 1)];
digitdata = [digitdata; x7]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(x7,1), 1)];
digitdata = [digitdata; x8]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(x8,1), 1)];
digitdata = [digitdata; x9]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(x9,1), 1)];
digitdata = digitdata/255;

totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata,2);
batchsize = 100;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets x0 x1 x2 x3 x4 x5 x6 x7 x8 x9;

%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%
xt0=dlmread('E:\workspace\MLProject\0t.txt',',');
xt1=dlmread('E:\workspace\MLProject\1t.txt',',');
xt2=dlmread('E:\workspace\MLProject\2t.txt',',');
xt3=dlmread('E:\workspace\MLProject\3t.txt',',');
xt4=dlmread('E:\workspace\MLProject\4t.txt',',');
xt5=dlmread('E:\workspace\MLProject\5t.txt',',');
xt6=dlmread('E:\workspace\MLProject\6t.txt',',');
xt7=dlmread('E:\workspace\MLProject\7t.txt',',');
xt8=dlmread('E:\workspace\MLProject\8t.txt',',');
xt9=dlmread('E:\workspace\MLProject\9t.txt',',');

digitdata=[];
targets=[];
digitdata = [digitdata; xt0]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(xt0,1), 1)]; 
digitdata = [digitdata; xt1]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(xt1,1), 1)]; 
digitdata = [digitdata; xt2]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(xt2,1), 1)];
digitdata = [digitdata; xt3]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(xt3,1), 1)];
digitdata = [digitdata; xt4]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(xt4,1), 1)];
digitdata = [digitdata; xt5]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(xt5,1), 1)];
digitdata = [digitdata; xt6]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(xt6,1), 1)];
digitdata = [digitdata; xt7]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(xt7,1), 1)];
digitdata = [digitdata; xt8]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(xt8,1), 1)];
digitdata = [digitdata; xt9]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(xt9,1), 1)];
digitdata = digitdata/255;

totnum=size(digitdata,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0);
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets xt0 xt1 xt2 xt3 xt4 xt5 xt6 xt7 xt8 xt9;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 
