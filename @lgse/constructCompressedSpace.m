function constructCompressedSpace(this, kernels, iteration_number)
  this.compressedTrainPoints = zeros(this.sampleSize, this.reducedDimension);
  pseudoInverseH = cell(this.sampleSize, 1);
  for pointIndex = 1:this.sampleSize
    meanCentered = zeros(1, this.originalDimension);
    pseudoInverseH{pointIndex} = zeros(this.originalDimension, this.reducedDimension);
    for comparePoint = 1:this.sampleSize
      meanCentered = meanCentered  + kernels(pointIndex, comparePoint) * ...
        (this.trainPoints(pointIndex, :) - this.trainPoints(comparePoint, :));
      pseudoInverseH{pointIndex} = pseudoInverseH{pointIndex} + kernels(pointIndex, comparePoint) * ...
        (this.projectionJacobians{comparePoint} + this.projectionJacobians{pointIndex});
    end
    [Q, R] = qr(pseudoInverseH{pointIndex}, 0);
    pseudoInverseH{pointIndex} = (R^(-1) * Q')';
    meanCentered = meanCentered / sum(kernels(:, pointIndex));
    this.compressedTrainPoints(pointIndex, :) = 2 * meanCentered * pseudoInverseH{pointIndex};
  end
  initialCompressedPoints = this.compressedTrainPoints;
%   for dimensionIndex = 1:this.reducedDimension
%     this.compressedTrainPoints(:, dimensionIndex) = ...
%       this.compressedTrainPoints(:, dimensionIndex) - mean(this.compressedTrainPoints(:, dimensionIndex));
%   end
  this.calculateDeltaCompression(kernels);
  historyDeltaCompression = [this.currentDelta];
  disp('_____________________');
  for iteration = 1:iteration_number
    if mod(iteration, 10) == 0
      disp(['Current iteration=', num2str(iteration)]);
    end
    newCompressedPoints = zeros(this.sampleSize, this.reducedDimension);
    newOriginalPoints = zeros(this.sampleSize, this.originalDimension);
    for currentPoint = 1:this.sampleSize
      for referencePoint = 1:this.sampleSize
        newOriginalPoints(currentPoint, :) = newOriginalPoints(currentPoint, :) + ...
          kernels(referencePoint, currentPoint) * this.compressedTrainPoints(referencePoint, :) * ...
          (this.projectionJacobians{comparePoint} + this.projectionJacobians{pointIndex})';
      end
      newCompressedPoints(currentPoint, :) = newOriginalPoints(currentPoint, :) * pseudoInverseH{pointIndex} + ...
        initialCompressedPoints(currentPoint, :);
    end
    this.compressedTrainPoints = newCompressedPoints;
%     for dimensionIndex = 1:this.reducedDimension
%       this.compressedTrainPoints(:, dimensionIndex) = ...
%         this.compressedTrainPoints(:, dimensionIndex) - mean(this.compressedTrainPoints(:, dimensionIndex));
%     end
    this.calculateDeltaCompression(kernels);
    historyDeltaCompression = [historyDeltaCompression this.currentDelta];
  end  
  this.historyDeltaCompression = historyDeltaCompression;
end