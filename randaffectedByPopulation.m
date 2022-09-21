function  [x] = randaffectedByPopulation(row,column,population)

z=rand(row,column);
z=z+population;
 for i = 1:row,
     for j=1:column
         if(z(i,j)>=1)
             z(i,j)=0.9;
         end
     end
 end
 x=z;
