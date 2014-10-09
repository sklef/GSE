% experiments with kernel width
trainSize = 500; % [100, 200]
kernelWidths = 0.25:0.25:0.75; % 2;
testSize = 1000;
rng(0);
surfaceNames = {'cylinder', 'saddle', 'ellipsoid'}; % ,  
methods = {'GSE', 'OGSE'};
% MSE = zeros(length(methods), length(kernelWidths), length(surfaceNames));
% metrics = cell(length(surfaceNames), length(kernelWidths), length(methods));
mkdir('pictures');
for surfaceIndex = 1:length(surfaceNames)
  disp(surfaceNames{surfaceIndex});
  surfaceName = surfaceNames{surfaceIndex};
  [testX, testTangentSpace, parametrizationTest] = ...
       generateSampleOnSurface(testSize, surfaceName);
  handle = figure();
  scatter3(testX(:, 1), testX(:, 2), testX(:, 3), [], parametrizationTest(:, 1), 'filled');
  saveas(handle, strcat(surfaceName, 'TestSample.png'));
  close(handle);
  trainSize = trainSizes(trainSizeIndex);
  [trainX, trainTangentSpace, parametrization] = ...
       generateSampleOnSurface(trainSize, surfaceName);
  for kernelWidthIndex = 1:length(kernelWidths)
    disp(kernelWidths(kernelWidthIndex));
    for methodIndex = 1:length(methods)
      isOldOptimization = 2 - methodIndex;
      mapping = lgse('EuclideanMetricsThreshold', 1, 'EigenValuesThreshold',...
                  Inf, 'CauchyBinetMetricsThreshold', 10, 'KernelWidth',...
                  kernelWidths(kernelWidthIndex),'oldWayOptimization', isOldOptimization);
      mapping.train(trainX, 2);
      reducedTestX = mapping.compress(testX);
      handle = figure();
      scatter(reducedTestX(1, :), reducedTestX(2, :), [], parametrizationTest(:, 1), 'filled');
      saveas(handle, strcat(surfaceName, 'Compressed-kernel', num2str(kernelWidths(kernelWidthIndex)), '-', methods{methodIndex}, '.png'));
      close(handle);
      recconstructedTestX = mapping.decompress(reducedTestX);
      MSE(methodIndex, kernelWidthIndex, surfaceIndex) = ...
        sqrt(trace((recconstructedTestX - testX) * (recconstructedTestX - testX)') / trainSize);
      save('MSE.mat', 'MSE');
      metrics{surfaceIndex, kernelWidthIndex, methodIndex} = ...
        calculateLocalIsometry(testX, reducedTestX', kernelWidths(kernelWidthIndex), mapping.EuclideanMetricsThreshold);
      save('metrics.mat', 'metrics');
      randomPairsNumber = 1000;
      firstPoints = randi(testSize, randomPairsNumber);
      secondPoints = randi(testSize, randomPairsNumber);
      for pair = 1:randomPairsNumber
        distance(pair) = sqrt(sum((testX(firstPoints(pair), :) - testX(secondPoints(pair), :)) .^ 2));
        distanceReduced(pair) = sqrt(sum((reducedTestX(:, firstPoints(pair)) - reducedTestX(:, secondPoints(pair))) .^ 2));
      end
      handle = figure();
      scatter(distance, distanceReduced, '*r');
      hold on
      plot([0, max(max(distance), max(distanceReduced))], [0, max(max(distance), max(distanceReduced))], 'b');
      saveas(handle, strcat('pictures', filesep, surfaceName, '-Distances-KW-', num2str(kernelWidths(kernelWidthIndex)), methods{methodIndex}, '.png'));
      close(handle);
    end    
  end
  %%
  for errorIndex = 1:3
    handle = figure();
    hold on
    GSEMetricToPlot = zeros(1, length(kernelWidths));
    OGSEMetricToPlot = GSEMetricToPlot;
    for kernelWidthIndex = 1:length(kernelWidths)
      GSEMetricToPlot(kernelWidthIndex) = metrics{surfaceIndex, kernelWidthIndex, 1}(errorIndex, 1);
      OGSEMetricToPlot(kernelWidthIndex) = metrics{surfaceIndex, kernelWidthIndex, 2}(errorIndex, 1);
    end
    plot(kernelWidths, GSEMetricToPlot, '-*r');
    plot(kernelWidths, OGSEMetricToPlot, '-vb');
    saveas(handle, strcat('pictures', filesep, num2str(errorIndex), surfaceName, '-Metrics-KW', '.png'));
    close(handle);
  end
end