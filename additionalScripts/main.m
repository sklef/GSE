trainSize = 200;
rng(0);
[trainX, trainTangentSpace, parametrization] = ...
    generateSampleOnSurface(trainSize, 'ellipsoid');
mapping = lgse('EuclideanMetricsThreshold', 10, 'EigenValuesThreshold',...
                Inf, 'CauchyBinetMetricsThreshold', 0.2, 'KernelWidth',...
                1,'oldWayOptimization', false);
            
[trainX, trainTangentSpace, parametrization] = ...
    generateSampleOnSurface(trainSize, 'ellipsoid');
mapping.train(trainX, 2);
reducedTrainX = mapping.compressedTrainPoints
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
