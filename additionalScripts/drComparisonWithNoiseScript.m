fun = @(u, v)[u.*cos(u), v, u.*sin(u)]; % swissroll
minT = [3*pi/2, 0];
rangeT = [3*pi, 21];
dim = 2;

% fun = @(t)[t.*cos(t),t.*sin(t), t];
% minT = [3*pi/2];
% rangeT = [3*pi];
% dim = 1;

methods = {'Isomap', 'LLE', 'HLLE', 'LTSA', 'LGSE'}; %{'LGSE'};% {'Isomap', 'LLE', 'HLLE', 'LTSA', 'LGSE'};

seeds = 1:10;
noises = 0:0.01:0.1;
trainSizes = 800;
testSize = 1000;
nns = 7:14; %7:16; %0.3:0.05:0.55;
kws = 0.005;%2.^(-2:-2:-10);

% trainErr1 = Inf(length(seeds), length(trainSizes), length(nns), length(kws), length(methods));
% trainErr2 = Inf(length(seeds), length(trainSizes), length(nns), length(kws), length(methods));
% testErr = Inf(length(seeds), length(trainSizes), length(nns), length(kws), length(methods));

trainErr1 = NaN(length(seeds), length(noises), length(nns), length(methods));
trainErr2 = NaN(length(seeds),length(noises), length(nns), length(methods));
testErr = NaN(length(seeds),length(noises), length(nns), length(methods));

wb = waitbar(0, 'Calculating...');
for seed = seeds
  for noiseIdx = 1:length(noises)
    noiseLevel = noises(noiseIdx);
    for sampleSizeIdx = 1:length(trainSizes)
      sampleSize = trainSizes(sampleSizeIdx);
      rng(seed);
      trainT = rand(sampleSize, dim);
      testT = rand(testSize, dim);
      [col, idx] = sort(trainT(:,1));
      trainT = trainT(idx, :) .* repmat(rangeT, sampleSize, 1) + repmat(minT, sampleSize, 1);
      [colTest, idx] = sort(testT(:,1));
      testT = testT(idx, :) .* repmat(rangeT, testSize, 1) + repmat(minT, testSize, 1);
      
      trainX = fun(trainT(:,1),trainT(:,2));
      
      [trainX, ps] = mapminmax(trainX');
      
      trainX = trainX' + noiseLevel * randn(size(trainX'));
      testX = fun(testT(:,1),testT(:,2));
      testX = mapminmax('apply', testX',ps)';
      %     trainX = fun(trainT);
      %     testX = fun(testT);
      
      for methodIdx = 1:length(methods)-1;
        method = methods{methodIdx};
        for nnIdx = 1:length(nns)
          for kwIdx = 1:length(kws)
            kw = kws(kwIdx);
            nn = nns(nnIdx);
            try
              if ~strcmp(method, 'LGSE')
                tic;[trainT1, mapping] = compute_mapping(trainX, method, dim, nn);toc
              else
                %                                 options.oldWayOptimization = false;
                %
                %                                 options.euclideanMetricsThreshold = nn; % epsilon_0
                %                                 options.eigenValuesThreshold = Inf; % epsilon_1
                %                                 options.cauchyBinetMetricsThreshold = 1; % epsilon_2
                %                                 options.kernelWidth = kw;
                
                %             mapping = lgse(options);
                mapping = lgse('EuclideanMetricsThreshold', nn, 'EigenValuesThreshold', Inf, 'CauchyBinetMetricsThreshold', 1, 'KernelWidth', kw, 'oldWayOptimization', false);
                tic;mapping.train(trainX, dim);toc
              end
            catch e
              disp(e.getReport)
              continue
            end
            
            if ~strcmp(method, 'LGSE') && ~strcmp(method, 'LTSA')&& ~strcmp(method, 'HLLE')
              recX1 = decompressPoints(trainT1, trainT1, trainX, nn);
              trainT2 = out_of_sample(trainX, mapping);
              recX2 = decompressPoints(trainT2, trainT1, trainX, nn);
              testT = out_of_sample(testX, mapping);
              recX3 = decompressPoints(testT, trainT1, trainX, nn);
            elseif strcmp(method, 'LGSE')
              recX1 = mapping.decompress(mapping.compressedTrainPoints);
              trainT2 = mapping.compress(trainX);
              recX2 = mapping.decompress(trainT2);
              testT = mapping.compress(testX);
              recX3 = mapping.decompress(testT);
            else
              recX1 = decompressPoints(trainT1, trainT1, trainX, nn);
              trainT2 = out_of_sample_est(trainX, trainX, trainT1);
              recX2 = decompressPoints(trainT2, trainT1, trainX, nn);
              testT = out_of_sample_est(testX, trainX, trainT1);
              recX3 = decompressPoints(testT, trainT1, trainX, nn);
            end
            
            if isfield(mapping, 'conn_comp')
              cc = mapping.conn_comp;
              [trainErr1(seed, noiseIdx, nnIdx, methodIdx), trainErr2(seed, noiseIdx, nnIdx, methodIdx), ...
                testErr(seed, noiseIdx, nnIdx, methodIdx)] = calcMappingErrs(trainX, recX1, recX2, testX, recX3, cc);
            else
              [trainErr1(seed, noiseIdx, nnIdx, methodIdx), trainErr2(seed, noiseIdx, nnIdx, methodIdx), ...
                testErr(seed, noiseIdx, nnIdx, methodIdx)] = calcMappingErrs(trainX, recX1, recX2, testX, recX3);
            end
            %         end
          end
        end
      end
    end
  end
  waitbar(seed/length(seeds), wb);
  if ~strcmp(method, 'LGSE')
    save noise
  else
    save lgseNoise
  end
end

close(wb)
