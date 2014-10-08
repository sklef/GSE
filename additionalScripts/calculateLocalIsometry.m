function metrics = calculateLocalIsometry(points, compressedPoints, kernelWidths, euclideanMetricsThreshold)
  numberPoints = size(points, 1);
  if nargin < 4
    disp('euclideanMetricsThreshold in calculateLocalIsometry is unset!');
    euclideanMetricsThreshold = 1000;
  end
  metrics = zeros(3, length(kernelWidths));
  kernels = calculateKernelsFunction(points, kernelWidths, points, euclideanMetricsThreshold);
  meanWeightedPointsDistance = zeros(1, length(kernelWidths));
  for point1 = 1:numberPoints
    for point2 = 1:numberPoints
      pointsDistance = sqrt(sum((points(point1, :) - points(point2, :)) .^ 2));
      compressedPointsDistance = sqrt(sum((compressedPoints(point1, :) - compressedPoints(point2, :)) .^ 2));
      metrics(1, :) = metrics(1, :) + reshape(kernels(point1, point2, :), 1, []) * ...
         compressedPointsDistance / (pointsDistance + eps);
      metrics(2, :) = metrics(2, :) + reshape(kernels(point1, point2, :), 1, []) * ...
         abs(pointsDistance - compressedPointsDistance); 
      metrics(3, :) = metrics(3, :) + reshape(kernels(point1, point2, :), 1, []) * ...
         2 * compressedPointsDistance / (pointsDistance + compressedPointsDistance + eps);
      meanWeightedPointsDistance = meanWeightedPointsDistance + reshape(kernels(point1, point2, :), 1, []) * pointsDistance;
    end
  end
  metrics(1, :) = metrics(1, :) ./ reshape(sum(sum(kernels, 1), 2), 1, []);
  metrics(3, :) = metrics(3, :) ./ reshape(sum(sum(kernels, 1), 2), 1, []);
  metrics(2, :) = metrics(2, :) ./ meanWeightedPointsDistance;
end

