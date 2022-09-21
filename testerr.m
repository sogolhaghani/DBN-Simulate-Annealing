function  [err] = testerr(testbatchdata,testbatchtargets,vishid_l0,hidbiases_l0,vishid,visbiases,hidbiases,labhid,labbiases)

[numdim, numhid]=size(vishid_l0);
[numcases, ~, numbatches]=size(testbatchdata);
counter=zeros(10,10000);
%fprintf(1,'1 %d\n',size(counter));
targets_all = zeros(10000,10);
%fprintf(1,'2 %d\n',size(targets_all));
for batch=1:100
 targets_all( (batch-1)*100+1:batch*100,:) = testbatchtargets(:,:,batch);
end
%fprintf(1,'3 %d\n',size(targets_all));
  bias_hid_l0= repmat(2*hidbiases_l0,numcases,1);
  bias_pen = repmat(hidbiases,numcases,1);

for batch= 1:100
     inter = zeros(numcases,10);
     data = testbatchdata(:,:,batch); 

     totin_h1 = data*(2*vishid_l0) + bias_hid_l0;
     temp_h1 = 1./(1 + exp(-totin_h1));


      for tt=1:10
        targets = zeros(numcases,10);
        targets(:,tt)=1; 
        lab_bias =  targets*labhid; 

        temp1 = temp_h1*visbiases' + targets*labbiases';   
        prod_3 = ones(numcases,1)*hidbiases + (temp_h1*vishid + targets*labhid);
        p_vl  = temp1 + sum(log(1+exp(prod_3)),2);
        inter(:,tt) = p_vl;  
      end 
     
    counter(:,(batch-1)*100+1:batch*100)=inter';
end 
 %fprintf(1,'4 %d\n',size(counter));
  [t, J]=max(counter',[],2);
   [t1, J1]=max(targets_all,[],2);
  % fprintf(1,'t %d\n',t);
  % fprintf(1,'t1 %d\n',t1);
   err1=length(find(J~=J1));
 % fprintf(1,'err %d\n',err1);
 err = err1;

