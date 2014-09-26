function calculateJacobianComponent(this, dimensionIndex, kernels, iteration_number)
projectionJacobiansNew = this.projectionJacobians;
bestProjectionJacobians = this.projectionJacobians;
% disp('_____________________');
numberMultistartIterations = 1; % 5
this.calculateDelta(kernels, dimensionIndex);
bestDelta = this.currentDelta;
historyDelta = cell(numberMultistartIterations, 1);
if dimensionIndex == this.reducedDimension
  numberMultistartIterations = 1;
end
for multistartIteration = 1:numberMultistartIterations
  % disp(['multistartIteration=', num2str(multistartIteration)]);
  for pointIndex = 1:this.sampleSize
    this.projectionJacobians{pointIndex}(:, dimensionIndex:end) = this.localPCs{pointIndex}(:, dimensionIndex:end) * ...
      orth(rand(this.reducedDimension - dimensionIndex + 1));
  end
%   if dimensionIndex == 1
%     disp(this.projectionJacobians{pointIndex}(:, 1));
%   end
  this.calculateDelta(kernels, dimensionIndex);
  historyDelta{multistartIteration} = [this.currentDelta];
  for iteration = 1:iteration_number
  
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
    historyDelta{multistartIteration} = [historyDelta{multistartIteration}, this.currentDelta];
  end
  if this.currentDelta < bestDelta
    bestDelta = this.currentDelta;
    bestProjectionJacobians = this.projectionJacobians;
  end
end
this.historyDelta{dimensionIndex} = historyDelta;
this.projectionJacobians = bestProjectionJacobians;
end

%%
%% find good rotation 
%   this.projectionJacobians = projectionJacobiansNew;
%   this.updatePCs(dimensionIndex);
%   this.updateVs(dimensionIndex);
%   rotationDimencionality = this.reducedDimension - dimensionIndex + 1;
%   G = zeros(rotationDimencionality);
%   for point = 1:this.sampleSize
%     for comparePoint = 1:this.sampleSize
%       G = G + 2 * kernels(point, comparePoint) * ...
%         (projectionJacobiansNew{point}(:, dimensionIndex) - projectionJacobiansNew{comparePoint}(:, dimensionIndex))' * ...
%         (projectionJacobiansNew{point}(:, dimensionIndex) - projectionJacobiansNew{comparePoint}(:, dimensionIndex));
%     end
%   end
%   G = G / sum(sum(kernels, 1), 2);
%   A = G - G';
%   Q = eye(rotationDimencionality);
%   rotationIterations = 10;
%   for iterationForRotation = 1:rotationIterations
%     newQ = @(tau)((eye(rotationDimencionality) + tau * A)^(-1) * (eye(rotationDimencionality) - tau * A));
%     functionToMinimize = @(tau)(trace(newQ(tau) * Q * G * Q' * newQ(tau)'));
%     tauBest = fminbnd(functionToMinimize, -1/2 * norm(G), 1/2 * norm(G));
%     Q = newQ(tauBest);
%   end
%   for pointIndex = 1:this.sampleSize
%     projectionJacobiansNew{pointIndex}(:, dimensionIndex) = ...
%       this.localPCs{pointIndex} * this.vs{pointIndex} * Q;
%   end
%   this.projectionJacobians = projectionJacobiansNew;
%   this.updatePCs(dimensionIndex);
%   this.updateVs(dimensionIndex);
%   this.calculateDelta(kernels, dimensionIndex);
%   historyDelta = [historyDelta, this.currentDelta];
%   for pointIndex = 1:this.sampleSize
%     projectionJacobiansNew{pointIndex}(dimensionIndex:end, dimensionIndex) = ...
%       projectionJacobiansNew{pointIndex}(dimensionIndex:end, dimensionIndex) / normalization_coefficient_global;
%   end
