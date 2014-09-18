function calculateDeltaCompression(this, kernels)
% calculates Delta_n(h)
  delta = 0;
  for firstPointIndex = 1:length(this.trainPoints)
    for secondPointIndex = 1:(firstPointIndex - 1)
      delta = delta + kernels(firstPointIndex, secondPointIndex) * ...
        norm(this.trainPoints(firstPointIndex, :) - this.trainPoints(secondPointIndex, :) - ...
        (this.compressedTrainPoints(firstPointIndex, :) - this.compressedTrainPoints(secondPointIndex, :)) * ...
        this.projectionJacobians{firstPointIndex}')^2; 
    end
  end
  this.currentDelta = delta;
end