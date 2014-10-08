function kernels = calculateKernelsFunction(points, kernelWidth, otherPoints, euclideanMetricsThreshold)
% Example:
% points = rand(10, 3);
% kernelWidth = [0.1, 1, 10];
% otherPoints = rand(12, 3);
% euclideanMetricsThreshold = 1;
% kernels = calculateKernels(points, kernelWidth, otherPoints, euclideanMetricsThreshold)
if nargin < 3
  otherPoints = points;
end
if nargin < 4
  euclideanMetricsThreshold = 1000;
end
numPoints = size(points, 1);
numOtherPoints = size(otherPoints, 1);
kernels = zeros(numPoints, numOtherPoints, length(kernelWidth));

for pointIndex1 = 1:numPoints
  for pointIndex2 = 1:numOtherPoints
    euclideanDistance = sqrt(sum((points(pointIndex1, :) - otherPoints(pointIndex2, :)) .^ 2));
    if euclideanDistance < euclideanMetricsThreshold
      kernels(pointIndex1, pointIndex2, :) = exp(-kernelWidth * euclideanDistance^2);
    end
  end
end

end