function calculateJacobianComponent(this, dimensionIndex, kernels, iteration_number)
projectionJacobiansNew = this.projectionJacobians;
this.calculateDelta(kernels, dimensionIndex);
historyDelta = [this.currentDelta];
% disp('_____________________');
for iteration = 1:iteration_number
%   if mod(iteration, 10) == 0
%     disp(['Current iteration=', num2str(iteration)]);
%   end
  for currentPoint = 1:this.sampleSize
    projectionJacobiansNew{currentPoint}(:, dimensionIndex) = projectionJacobiansNew{currentPoint}(:, dimensionIndex) * 0;
    normalization_coefficient = sum(kernels(:, currentPoint));
    projection_matrix = this.localPCs{currentPoint}(:, dimensionIndex:end)  * this.localPCs{currentPoint}(:, dimensionIndex:end)';
    for referencePoint = 1:this.sampleSize
      projectionJacobiansNew{currentPoint}(:, dimensionIndex) = projectionJacobiansNew{currentPoint}(:, dimensionIndex) + ...
        kernels(currentPoint, referencePoint) * ...
          this.projectionJacobians{referencePoint}(:, dimensionIndex) / normalization_coefficient;
    end
    projectionJacobiansNew{currentPoint}(:, dimensionIndex) = projection_matrix * projectionJacobiansNew{currentPoint}(:, dimensionIndex);
  end
  normalization_coefficient_global = 0;
  for pointIndex = 1:this.sampleSize
    normalization_coefficient_global = normalization_coefficient_global + sum(kernels(:, pointIndex)) * ...
      norm(this.projectionJacobians{pointIndex}(:,dimensionIndex), 'fro');
  end
  % disp(['Normalization coefficient=', num2str(normalization_coefficient_global)]);
  for pointIndex = 1:this.sampleSize
    projectionJacobiansNew{pointIndex}(:, dimensionIndex) = ...
      projectionJacobiansNew{pointIndex}(:, dimensionIndex) / normalization_coefficient_global;
  end
%   normalization_coefficient_global = 0;
%   for pointIndex = 1:this.sampleSize
%     normalization_coefficient_global = normalization_coefficient_global + sum(kernels(:, pointIndex)) * ...
%       norm(this.projectionJacobians{pointIndex}(:,dimensionIndex), 'fro');
%   end
%   disp(['Solution Norm=', num2str(normalization_coefficient_global)]);
  this.projectionJacobians = projectionJacobiansNew;
  this.calculateDelta(kernels, dimensionIndex);
  historyDelta = [historyDelta, this.currentDelta];
end  
this.historyDelta{dimensionIndex} = historyDelta;

end
