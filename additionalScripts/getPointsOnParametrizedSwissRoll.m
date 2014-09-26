function [dataSet, pointsParameters] = getPointsOnParametrizedSwissRoll(inputPoints, parameters)
   dataSet = zeros(length(inputPoints(:,1)),3);
   pointsParameters= zeros(length(inputPoints(:,1)),2);
   if parameters.alpha ~= 0
      kParam = (2 * pi * sqrt(1 + (2 * pi) ^ 2) + log(2 * pi + sqrt(1 + (2 * pi) ^ 2))) / (2 * pi * parameters.alpha * sqrt(1 + (2 * pi * parameters.alpha) ^ 2) + log(2 * pi * parameters.alpha + sqrt(1 + (2 * pi * parameters.alpha) ^ 2)));
      for i = 1:length(inputPoints(:,1))
         dataSet(i,:) = parameters.center + [kParam * parameters.alpha * inputPoints(i,1)*cos(parameters.alpha * inputPoints(i,1)), inputPoints(i,2), kParam * parameters.alpha * inputPoints(i,1)*sin(parameters.alpha * inputPoints(i,1))];
         pointsParameters(i,:) = [inputPoints(i,1), inputPoints(i,2)];
      end
   else
      kParam = (2* pi *sqrt(1 + (2 * pi) ^ 2) + log(2 * pi + sqrt(1 + (2 * pi) ^ 2)))/(2*2*pi);
      for i = 1:length(inputPoints(:,1))
         dataSet(i,:) = parameters.center + [kParam * inputPoints(i,1), inputPoints(i,2), 0];
         pointsParameters(i,:) = [inputPoints(i,1), inputPoints(i,2)];
      end      
   end    
end
