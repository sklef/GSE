function kernels = calculateKernels(model, points, weights, otherPoints)
if nargin < 4
  otherPoints = points;
end
numPoints = size(points, 2);
numOtherPoints = size(otherPoints, 2);
kernels = zeros(numPoints, numOtherPoints);
% prealloc = round(0.1 * max(numPoints, numOtherPoints)^2);
% next = 1;
% idx1 = zeros(prealloc, 1);
% idx2 = zeros(prealloc, 1);
% vals = zeros(prealloc, 1);
if nargin == 2
  for pointIndex1 = 1:numPoints
    for pointIndex2 = 1:numOtherPoints
      euclideanDistance = sqrt(sum((points(:,pointIndex1) - otherPoints(:,pointIndex2)) .^ 2));
      if euclideanDistance < model.EuclideanMetricsThreshold
%         idx1(next) = pointIndex1;
%         idx2(next) = pointIndex2;
%         vals(next) = exp(-model.kernelWidth.*euclideanDistance^2);
%         next = next + 1;
        kernels(pointIndex1,pointIndex2) = exp(-model.KernelWidth.*euclideanDistance^2);
      end
    end
  end
else
  for pointIndex1 = 1:numPoints
    for pointIndex2 = 1:numOtherPoints
      euclideanDistance = sqrt(sum((weights{pointIndex2}*(points(:,pointIndex1) - otherPoints(:,pointIndex2))) .^ 2));
      if euclideanDistance < model.EuclideanMetricsThreshold
%         idx1(next) = pointIndex1;
%         idx2(next) = pointIndex2;
%         vals(next) = exp(-model.kernelWidth.*euclideanDistance^2);
%         next = next + 1;
        kernels(pointIndex1,pointIndex2) = exp(-model.KernelWidth.*euclideanDistance^2);
      end
    end
  end
end
% idx1(next:end) = [];
% idx2(next:end) = [];
% vals(next:end) = [];

% kernels = sparse(idx1, idx2, vals, numPoints, numOtherPoints);
end