fprintf(1,'Converting Raw files into Matlab format \n');
mymakebatch;
fprintf(1,'Pretraining a Deep Belief Network. \n');
[numcases numdims numbatches]=size(batchdata);
% Training 1st layer %
numhid=500; maxepoch=50;
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm
% Training 2st layer %
close all 
numpen = 1000; 
maxepoch=50;
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
restart=1;
mymakebatch;
rbm_l2



