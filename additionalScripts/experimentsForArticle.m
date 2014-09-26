trainSizes = [250, 500, 1000, 2000]; % [100, 200]
testSize = 10000;
rng(0);
surfaceNames = {'saddle', 'ellipsoid',  'cylinder'};
methods = {'GSE', 'OGSE'};
MSE = zeros(length(methods), length(trainSizes), length(surfaceNames));
for surfaceIndex = 1:length(surfaceNames)
  disp(surfaceNames{surfaceIndex});
  surfaceName = surfaceNames{surfaceIndex};
  [testX, testTangentSpace, parametrizationTest] = ...
       generateSampleOnSurface(testSize, surfaceName);
  handle = figure();
  scatter3(testX(:, 1), testX(:, 2), testX(:, 3), [], parametrizationTest(:, 1), 'filled');
  saveas(handle, strcat(surfaceName, 'TestSample.png'));
  close(handle);
  for trainSizeIndex = 1:length(trainSizes)
    disp(trainSizes(trainSizeIndex));
    trainSize = trainSizes(trainSizeIndex);
    [trainX, trainTangentSpace, parametrization] = ...
       generateSampleOnSurface(trainSize, surfaceName);
    for methodIndex = 1:length(methods)
      isOldOptimization = 2 - methodIndex;
      mapping = lgse('EuclideanMetricsThreshold', 1, 'EigenValuesThreshold',...
                  Inf, 'CauchyBinetMetricsThreshold', 10, 'KernelWidth',...
                  1,'oldWayOptimization', isOldOptimization);
      mapping.train(trainX, 2);
      reducedTestX = mapping.compress(testX);
      handle = figure();
      scatter(reducedTestX(1, :), reducedTestX(2, :), [], parametrizationTest(:, 1), 'filled');
      saveas(handle, strcat(surfaceName, 'CompressedTestSample', num2str(trainSizes(trainSizeIndex)), methods{methodIndex}, '.png'));
      close(handle);
      recconstructedTestX = mapping.decompress(reducedTestX);
      MSE(methodIndex, trainSizeIndex, surfaceIndex) = ...
        sqrt(trace((recconstructedTestX - testX) * (recconstructedTestX - testX)') / trainSize);
      save('MSE.mat', 'MSE');
    end
  end
end