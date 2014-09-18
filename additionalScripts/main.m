trainSize = 500;
rng(0);
[trainX, trainTangentSpace, parametrization] = ...
    generateSampleOnSurface(trainSize, 'ellipsoid'); % 'cylinder'
% %% tmp to get smaller part 
% indexes = trainX(:, 3) > 0.5;
% trainX = trainX(indexes, :);  
% parametrization = parametrization(indexes, :); 
% trainSize = size(trainX, 1);
% %% 
mapping = lgse('EuclideanMetricsThreshold', 0.5, 'EigenValuesThreshold',...
                Inf, 'CauchyBinetMetricsThreshold', 10, 'KernelWidth',...
                1,'oldWayOptimization', false);
% [trainX, indexes] = sortrows(trainX);
% parametrization = parametrization(indexes, :);
tmp = trainTangentSpace;
for pointIndex = 1:trainSize
  trainTangentSpace{pointIndex} = tmp{indexes(pointIndex)};
end
clear tmp
mapping.train(trainX, 2);
reducedTrainX = mapping.compressedTrainPoints;
figure()
scatter(reducedTrainX(1, :), reducedTrainX(2, :), [], parametrization(:, 1), 'filled');

%%
currentJacobians = [];
currentJacobiansNorms = [];
for point = 1:mapping.sampleSize
  currentJacobians = [currentJacobians; mapping.projectionJacobians{point}];
  currentJacobiansNorms = [currentJacobiansNorms; ...
    [norm(mapping.projectionJacobians{point}(:, 1)), norm(mapping.projectionJacobians{point}(:, 2))]];
end
figure()
plot(1:mapping.sampleSize, sort(log10(currentJacobiansNorms(:, 1))), '-*r');
hold on
plot(1:mapping.sampleSize, sort(log10(currentJacobiansNorms(:, 2))), '-*b');
figure()
mapping.plotProjectionJacobians()
