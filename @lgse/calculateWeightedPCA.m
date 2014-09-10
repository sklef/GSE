function [principalComponentsPerPoint, eigenValuesPerPoint] = calculateWeightedPCA(this, points, weights, otherPoints)
if nargin < 4
  otherPoints = points;
end

principalComponentsPerPoint = cell(size(weights, 1), 1);
eigenValuesPerPoint = cell(size(weights, 1), 1);

for pointIndex = 1:length(principalComponentsPerPoint)
  localWeights = weights(pointIndex, :)';
  localIndeces = localWeights > 0;
  localWeights = localWeights(localIndeces);
  localWeightedPoints = otherPoints(localIndeces,:).*repmat(sqrt(localWeights), 1, size(points, 2));
%   principalComponents = princomp(localWeightedPoints);
  localWeightedPoints = localWeightedPoints - repmat(mean(localWeightedPoints), size(localWeightedPoints,1),1);
%   principalComponentsPerPoint{pointIndex} = principalComponents(:,1:model.reducedDimension);
  [principalComponentsPerPoint{pointIndex}, eigenValuesPerPoint{pointIndex}] = eigs(localWeightedPoints' * localWeightedPoints, this.reducedDimension, 'LA');
end

end