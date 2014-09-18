function train(this, trainPoints, reducedDimension)
% LGSE train
this.iLogger.info('Training started.')
this.reducedDimension = reducedDimension;
[this.sampleSize, this.originalDimension] = size(trainPoints);
this.iLogger.debug('Sample size: %d', this.sampleSize);
this.iLogger.debug('Input dimension: %d', this.originalDimension);
this.iLogger.debug('Reduced dimension: %d', this.reducedDimension);

this.projectionJacobians = cell(this.sampleSize, 1);
this.vs = cell(this.sampleSize, 1);
vTv = cell(this.sampleSize, 1);

%% Preprocessing

this.iLogger.info('Preprocessing: scaling input space')
[trainPoints, this.mappingSettingsOriginalDimension] = mapminmax(trainPoints');
trainPoints = trainPoints';
this.trainPoints = trainPoints;

this.iLogger.info('Preprocessing: calculating kernels')

this.kernels = this.calculateKernels(trainPoints');

if this.iLogger.level < logLevel.Info
  this.iLogger.debug('Min number of neighbors: %d', min(sum(this.kernels ~= 0)));
  this.iLogger.debug('Maximum number of neighbors: %d', max(sum(this.kernels ~= 0)));
end

this.iLogger.info('Preprocessing: calculating tangent spaces')
% Weighted PCA
if this.newNormalization
  [this.localPCs, this.localEigenVals] = this.calculateWeightedPCA(trainPoints, this.kernels);  % Q(X_i), \Lambda(X_i)
else
  this.localPCs = this.calculateWeightedPCA(trainPoints, this.kernels);  % Q(X_i)
end

this.iLogger.info('Preprocessing: adjusting kernels')
% Adjusting kernels
[this.kernels, linearSpacesProjections] = this.adjustKernels(this.kernels, this.localPCs); %K_1(X_i, X_j) and S(X_i, X_j)
% [this.kernels, linearSpacesProjections] = getKernels(trainPoints, reducedDimension, this.KernelWidth, this.EuclideanMetricsThreshold, this.CauchyBinetMetricsThreshold);

if this.iLogger.level < logLevel.Info
  this.iLogger.debug('Min number of neighbors: %d', min(sum(this.kernels ~= 0)));
  this.iLogger.debug('Maximum number of neighbors: %d', max(sum(this.kernels ~= 0)));
end
if min(sum(this.kernels > 0)) < this.reducedDimension * 2
  disp('There are lonely points. Try to change parameters');
end
%% Compression Jacobian calculation
this.iLogger.info('Tangent space alignment')

weightedLinearSpacesProjections = cell(this.sampleSize);
diagonalLinearSpaceProjections = cell(this.sampleSize,1);
this.history = cell(this.sampleSize, 1);
diagonalLinearSpaceProjectionsEigenValues = cell(this.sampleSize,1);
eyeReducedDimension = eye(reducedDimension);

for pointIndex1 = 1:this.sampleSize
  for pointIndex2 = pointIndex1:this.sampleSize
    if pointIndex1 == pointIndex2
      weightedLinearSpacesProjections{pointIndex1, pointIndex2} = eyeReducedDimension;
    else
      if this.newNormalization
        weightedLinearSpacesProjections{pointIndex1, pointIndex2} = this.kernels(pointIndex1,pointIndex2) * ...
          this.localEigenVals{pointIndex1} * (linearSpacesProjections{pointIndex1, pointIndex2}) * this.localEigenVals{pointIndex2};
        weightedLinearSpacesProjections{pointIndex2, pointIndex1} = weightedLinearSpacesProjections{pointIndex1, pointIndex2}';
      else
        weightedLinearSpacesProjections{pointIndex1, pointIndex2} = this.kernels(pointIndex1,pointIndex2) * ...
          (linearSpacesProjections{pointIndex1, pointIndex2});
        weightedLinearSpacesProjections{pointIndex2, pointIndex1} = weightedLinearSpacesProjections{pointIndex1, pointIndex2}';
      end
    end
    
  end
  diagonalLinearSpaceProjections{pointIndex1} = eyeReducedDimension*sum(this.kernels(pointIndex1, :));
  if this.newNormalization
    diagonalLinearSpaceProjectionsEigenValues{pointIndex1} = eyeReducedDimension*sum(this.kernels(pointIndex1, :)) * this.localEigenVals{pointIndex1} ^ -2;
  end
end

phi1 = cell2mat(weightedLinearSpacesProjections);
phi0 = blkdiag(diagonalLinearSpaceProjections{:});
if this.newNormalization
  phi2 = blkdiag(diagonalLinearSpaceProjectionsEigenValues{:});
end

if this.oldWayOptimization
  % solving generalized eigenvalue problem
  options.disp = 0;
  options.isreal = 1;
  options.issym = 1;
  if this.newNormalization
    phi = phi2-phi1;
    phi0 = phi0/sum(sum(phi0));
    [eigenMatrix, ~] = eigs(phi, phi0, reducedDimension, 'SA', options);% W = {v_i}|i=1,n
  else
    phi = phi0-phi1;
    phi0 = phi0/sum(sum(phi0));
    [eigenMatrix, ~] = eigs(phi, phi0, reducedDimension, 'SA', options);% W = {v_i}|i=1,n
  end
  for pointIndex = 1:this.sampleSize
    v = eigenMatrix((pointIndex-1)*reducedDimension+1:pointIndex*reducedDimension,:); % v_i
    [U,~,V] = svd(v);
    this.vs{pointIndex} = U*V';
    if this.newNormalization
      vTv{pointIndex} = this.vs{pointIndex}' * this.localEigenVals{pointIndex} ^ 2 * this.vs{pointIndex};
      this.projectionJacobians{pointIndex} = this.localPCs{pointIndex} * this.localEigenVals{pointIndex} * this.vs{pointIndex}; % H(X_i)
    else
      this.projectionJacobians{pointIndex} = this.localPCs{pointIndex} * this.vs{pointIndex}; % H(X_i)
      vTv{pointIndex} = this.vs{pointIndex}' * this.vs{pointIndex};
   end
 end
else
    for index = 1:this.sampleSize
        this.projectionJacobians{index} = this.localPCs{index};
    end
    maxIterations = 10;
    for currentDim = 1:this.reducedDimension
      this.calculateJacobianComponent(currentDim, this.kernels, maxIterations);
      this.updatePCs(currentDim);
      this.updateVs(currentDim);
    end
    %% just to plot delta
    plot(0:maxIterations, log10(this.historyDelta{1}), '-*r')
    hold on
    plot(0:maxIterations, log10(this.historyDelta{2}), '-*b')
    xlabel('iteration')
    ylabel('log10 Delta')
    %%
    for pointIndex = 1:this.sampleSize
      vTv{pointIndex} = this.vs{pointIndex}' * this.vs{pointIndex};
      diagonalLinearSpaceProjections{pointIndex} = eyeReducedDimension*sum(this.kernels(pointIndex, :));
    end
end

%% Compression calculation
this.iLogger.info('Embedding')
% Solving linear system
LHS = cell(this.sampleSize);
RHS = cell(this.sampleSize, 1);

for pointIndex1 = 1:this.sampleSize
  RHStmp = zeros(this.reducedDimension, 1);
  LHSdiag = zeros(this.reducedDimension);
  for pointIndex2 = [1:pointIndex1-1 pointIndex1+1:this.sampleSize]
    tmp = this.kernels(pointIndex1, pointIndex2) * (vTv{pointIndex1} + vTv{pointIndex2});
    LHS{pointIndex1, pointIndex2} = tmp;
    LHSdiag = LHSdiag - tmp;
    RHStmp = RHStmp + this.kernels(pointIndex1, pointIndex2) * ...
      ((this.projectionJacobians{pointIndex1}' + this.projectionJacobians{pointIndex2}') * ...
      (trainPoints(pointIndex2,:) - trainPoints(pointIndex1,:))');
  end
  LHS{pointIndex1, pointIndex1} = LHSdiag;
  RHS{pointIndex1} = RHStmp;
end

% compressedPoints = reshape([cell2mat(LHS); repmat(eye(this.reducedDimension), 1, this.sampleSize)] \ ...
%   [cell2mat(RHS); zeros(this.reducedDimension, 1)], this.reducedDimension, this.sampleSize);


compressedPoints = reshape([cell2mat(LHS); cell2mat(diagonalLinearSpaceProjections)'] \ ...
  [cell2mat(RHS); zeros(this.reducedDimension, 1)], this.reducedDimension, this.sampleSize);
%% iterative approach 
% this.iLogger.info('Embedding')
% maxIterations = 100;
% this.constructCompressedSpace(this.kernels, maxIterations); 
% figure();
% plot(0:maxIterations, log10(this.historyDeltaCompression), '-*r')
% hold on
% xlabel('iteration')
% ylabel('log10 Delta(h)')

%% Postprocessing
[this.compressedTrainPoints, this.mappingSettingsReducedDimension] = mapminmax(compressedPoints); % this.compressedTrainPoints
% this.compressedTrainPoints = this.compressedTrainPoints';

% for pointIndex = 1:this.sampleSize
%   this.compressedTrainPoints(:, pointIndex) = pinv(this.vs{pointIndex1}) * this.compressedTrainPoints(:, pointIndex);
% end

this.iLogger.info('Training finished.')
end