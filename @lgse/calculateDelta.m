function calculateDelta(this, kernels, dimensionIndex)
% calculates Delta_n for a given dimension= dimensionIndex
  delta = 0;
  for firstPointIndex = 1:length(this.trainPoints)
    for secondPointIndex = 1:(firstPointIndex - 1)
      delta = delta + kernels(firstPointIndex, secondPointIndex) * ...
        norm(this.projectionJacobians{secondPointIndex}(:,dimensionIndex) - ...
          this.projectionJacobians{firstPointIndex}(:,dimensionIndex), 'fro');
    end
  end
  this.currentDelta = delta;
end

