if restart ==1,
  restart=0;

epsilonw      = 0.05;
epsilonvb     = 0.05;
epsilonhb     = 0.05;

CD=1;   
weightcost  = 0.001;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases, numdims, numbatches]=size(batchdata);
  epoch=1;

% Initializing weights and biases. 
  vishid     = 0.001*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

population=0;% My parameter
for epoch = epoch:maxepoch
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0; 
 for batch = 1:numbatches,
		 visbias = repmat(visbiases,numcases,1);
		 hidbias = repmat(2*hidbiases,numcases,1); 
	% START POSITIVE PHASE %
		  data = batchdata(:,:,batch);
		  data = data > randaffectedByPopulation(numcases,numdims,population);  

		  poshidprobs = 1./(1 + exp(-data*(2*vishid) - hidbias));    
		  batchposhidprobs(:,:,batch)=poshidprobs;
		  posprods    = data' * poshidprobs;
		  poshidact   = sum(poshidprobs);
		  posvisact = sum(data);
          % END OF POSITIVE PHASE  %

          % START NEGATIVE PHASE  %
		  poshidstates = poshidprobs > rand(numcases,numhid);
		  negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
		  negdata = negdata > randaffectedByPopulation(numcases,numdims,population); 
		  neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - hidbias));

		  negprods  = negdata'*neghidprobs;
		  neghidact = sum(neghidprobs);
		  negvisact = sum(negdata); 

		% END OF NEGATIVE PHASE %
		  err= sum(sum( (data-negdata).^2 ));
		  errsum = err + errsum;

		   if epoch>5,
			 momentum=finalmomentum;
		   else
			 momentum=initialmomentum;
		   end;

	% UPDATE WEIGHTS AND BIASES %
			vishidinc = momentum*vishidinc + ...
						epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
			visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
			hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

			vishid = vishid + vishidinc;
			visbiases = visbiases + visbiasinc;
			hidbiases = hidbiases + hidbiasinc;
		% END OF UPDATES % 
           population=population+(3/(60000*maxepoch));
		   if rem(batch,600)==0  
			 figure(1); 
			 dispims(negdata',28,28);
			 drawnow
		   end  
           
 end
  
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 

end;

%save vishid visbiases hidbiases epoch 


