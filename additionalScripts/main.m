trainSize = 1000;
rng(0);
%% points from generateSampleOnSurface
[trainX, trainTangentSpace, parametrization] = ...
    generateSampleOnSurface(trainSize, 'saddle'); % 'ellipsoid'  %  'cylinder' % 'ellipsoid' % 'sphere'
%% points from getPointsOnParametrizedSwissRoll
% parameters.center = [0 0 0];
% parameters.alpha = 2;
% inputPoints = rand(trainSize, 2);
% [trainX, parametrization] = getPointsOnParametrizedSwissRoll(inputPoints, parameters);
%
% %% tmp to get smaller part 
% indexes = parametrization(:, 1) < pi / 4;
% trainX = trainX(indexes, :);  
% parametrization = parametrization(indexes, :); 
% trainSize = size(trainX, 1);
%% 
%% trainX(:, 1) = (trainX(:, 1) - min(trainX(:, 1))).^2; % strange example
%%
mapping = lgse('EuclideanMetricsThreshold', 1, 'EigenValuesThreshold',...
                Inf, 'CauchyBinetMetricsThreshold', 10, 'KernelWidth',...
                1,'oldWayOptimization', false);
% [trainX, indexes] = sortrows(trainX);
% parametrization = parametrization(indexes, :);
tmp = trainTangentSpace;
% for pointIndex = 1:trainSize
%   trainTangentSpace{pointIndex} = tmp{indexes(pointIndex)};
% end
% clear tmp
mapping.train(trainX, 2);
reducedTrainX = mapping.compressedTrainPoints;
figure()
scatter(reducedTrainX(1, :), reducedTrainX(2, :), [], parametrization(:, 1), 'filled');
figure()
scatter3(trainX(:, 1), trainX(:, 2), trainX(:, 3), [], parametrization(:, 1), 'filled');
recconstructedPoints = mapping.decompress(mapping.compressedTrainPoints);
figure()
scatter3(recconstructedPoints(:, 1), recconstructedPoints(:, 2), recconstructedPoints(:, 3), [], parametrization(:, 1), 'filled');

%% try rotations
% mapping.calculateDelta(mapping.kernels, 1);
% delta = [mapping.currentDelta];
% bestDelta = mapping.currentDelta;
% bestProjectionJacobians = mapping.projectionJacobians;
% for i = 1:10
%   Q = orth(rand(2));
%   projectionJacobians = mapping.projectionJacobians;
%   for point = 1:mapping.sampleSize
%     projectionJacobians{point} = projectionJacobians{point} * Q;
%   end
%   mapping.setProjections(projectionJacobians);
%   mapping.calculateDelta(mapping.kernels, 1);
%   delta = [delta, mapping.currentDelta];
%   if mapping.currentDelta < bestDelta
%     bestDelta = mapping.currentDelta;
%     bestProjectionJacobians = mapping.projectionJacobians;
%   end
% end
% plot(delta, '-*');

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
