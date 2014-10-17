% experiments with kernel width
trainSizes = [250, 500, 1000, 2000]; % [100, 200]
kernelWidth = 1;
testSize = 1000;
rng(0);
surfaceNames = {'cone', 'cylinder', 'saddle', 'ellipsoid'}; % ,  
methods = {'GSE', 'OGSE'};
MSE = zeros(length(methods), length(trainSizes), length(surfaceNames));
metrics = cell(length(surfaceNames), length(trainSizes), length(methods));
mkdir('pictures');
for surfaceIndex = 1:length(surfaceNames)
  disp(surfaceNames{surfaceIndex});
  surfaceName = surfaceNames{surfaceIndex};
  [testX, testTangentSpace, parametrizationTest] = ...
       generateSampleOnSurface(testSize, surfaceName);
  handle = figure('Visible','Off');
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
      newNormalization = 2 - methodIndex;
      mapping = lgse('EuclideanMetricsThreshold', 1, 'EigenValuesThreshold',...
                  Inf, 'CauchyBinetMetricsThreshold', 10, 'KernelWidth',...
                  kernelWidth,'oldWayOptimization', isOldOptimization, 'newNormalization', newNormalization);
      mapping.train(trainX, 2);
      reducedTestX = mapping.compress(testX);
      handle = figure('Visible','Off');
      scatter(reducedTestX(1, :), reducedTestX(2, :), [], parametrizationTest(:, 1), 'filled');
      saveas(handle, strcat(surfaceName, 'Compressed-kernel', num2str(trainSizes(trainSizeIndex)), '-', methods{methodIndex}, '.png'));
      close(handle);
      recconstructedTestX = mapping.decompress(reducedTestX);
      MSE(methodIndex, trainSizeIndex, surfaceIndex) = ...
        sqrt(trace((recconstructedTestX - testX) * (recconstructedTestX - testX)') / trainSize);
      save('MSE.mat', 'MSE');
      metrics{surfaceIndex, trainSizeIndex, methodIndex} = ...
        calculateLocalIsometry(testX, reducedTestX', trainSizes(trainSizeIndex), mapping.EuclideanMetricsThreshold);
      save('metrics.mat', 'metrics');
      randomPairsNumber = 1000;
      firstPoints = randi(testSize, randomPairsNumber);
      secondPoints = randi(testSize, randomPairsNumber);
      for pair = 1:randomPairsNumber
        distance(pair) = sqrt(sum((testX(firstPoints(pair), :) - testX(secondPoints(pair), :)) .^ 2));
        distanceReduced(pair) = sqrt(sum((reducedTestX(:, firstPoints(pair)) - reducedTestX(:, secondPoints(pair))) .^ 2));
      end
      handle = figure('Visible','Off');
      scatter(distance, distanceReduced, '*r');
      hold on
      plot([0, max(max(distance), max(distanceReduced))], [0, max(max(distance), max(distanceReduced))], 'b');
      saveas(handle, strcat('pictures', filesep, surfaceName, '-Distances-KW-', num2str(trainSizes(trainSizeIndex)), methods{methodIndex}, '.png'));
      close(handle);
    end    
  end
  %%
  for errorIndex = 1:3
    handle = figure('Visible','Off');
    hold on
    GSEMetricToPlot = zeros(1, length(trainSizes));
    OGSEMetricToPlot = GSEMetricToPlot;
    for trainSizeIndex = 1:length(trainSizes)
      GSEMetricToPlot(trainSizeIndex) = metrics{surfaceIndex, trainSizeIndex, 1}(errorIndex, 1);
      OGSEMetricToPlot(trainSizeIndex) = metrics{surfaceIndex, trainSizeIndex, 2}(errorIndex, 1);
    end
    plot(trainSizes, GSEMetricToPlot, '-*r');
    plot(trainSizes, OGSEMetricToPlot, '-vb');
    saveas(handle, strcat('pictures', filesep, num2str(errorIndex), surfaceName, '-Metrics-KW', '.png'));
    close(handle);
  end
  handle = figure('Visible','Off');
  hold on
  GSEMetricToPlot = zeros(1, length(trainSizes));
  OGSEMetricToPlot = GSEMetricToPlot;
  for trainSizeIndex = 1:length(trainSizes)
    GSEMetricToPlot(trainSizeIndex) = metrics{surfaceIndex, trainSizeIndex, 1}(errorIndex, 1);
    OGSEMetricToPlot(trainSizeIndex) = metrics{surfaceIndex, trainSizeIndex, 2}(errorIndex, 1);
  end
  plot(trainSizes, GSEMetricToPlot, '-*r');
  plot(trainSizes, OGSEMetricToPlot, '-vb');
  saveas(handle, strcat('pictures', filesep, num2str(errorIndex), surfaceName, '-Metrics-KW', '.png'));
  close(handle);
end