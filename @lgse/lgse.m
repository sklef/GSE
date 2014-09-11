classdef lgse < handle
  
  properties (SetAccess = private)
    originalDimension
    reducedDimension
    sampleSize
    mappingSettingsOriginalDimension
    mappingSettingsReducedDimension
    
    trainPoints
    compressedTrainPoints
    
    kernels
    projectionJacobians
    vs
    localPCs
    localEigenVals
    
    
    EuclideanMetricsThreshold % \epsiolon_0
    EigenValuesThreshold % \epsilon_1
    CauchyBinetMetricsThreshold % \epsilon_2
    KernelWidth % \tau
    RegularizationLambda % \lambda_{orth}
    
    iLogger % log writer
    LoggingLevel % just for info
    
    % private parameters
    oldWayOptimization = false;
    newNormalization = false;
    strictOrthogonalization = false
  end
  
  properties (Constant)
    type = 'LGSE';
  end
  
  properties(SetAccess = public)
      bigVs = [];
  end
      
  
  methods (Access = 'public')
    
    function this = lgse(varargin)
      % Constructor for LGSE
      parser = inputParser; 
      parser.FunctionName = 'LGSE:Constructor';
      parser.CaseSensitive = true;
      parser.addOptional('LoggingLevel','Default');
      parser.addOptional('EuclideanMetricsThreshold', 0.5);
      parser.addOptional('EigenValuesThreshold', Inf);
      parser.addOptional('CauchyBinetMetricsThreshold', 0.5);
      parser.addOptional('KernelWidth', 0.01);
      parser.addOptional('RegularizationLambda', 1e3);
      parser.addOptional('oldWayOptimization', false);
      parser.addOptional('newNormalization', false);
      
      parser.parse(varargin{:})
      
      if strcmpi(parser.Results.LoggingLevel, 'Default')
        this.iLogger = logger(this.type);
      else
        this.iLogger = logger(this.type, parser.Results.LoggingLevel);
      end
      this.iLogger.info('Parameters Setting')
      setOptions = parser.Parameters;
      for idx = 1:length(setOptions)
        this.(setOptions{idx}) = parser.Results.(setOptions{idx});
        this.iLogger.info(strcat('Option "', setOptions{idx}, '"', ' set to "', num2str(parser.Results.(setOptions{idx})),'"'))
      end
      this.iLogger.info('Parameters Setting Complete')
    end
    
    function [reconstructedPoints, failedPoints] = reconstruct(model, points)
      [reconstructedPoints, failedPoints] = model.decompress(model.compress(points));
    end
    
    function setProjections(this, newProjections)
      this.projectionJacobians = newProjections;
    end
    
    constructCompressedSpace(model);
    
    train(model, points, reducedDimension);
   
    function [phi0, phi1, phi2] = reformulateProblem(this)

         [this.kernels, linearSpacesProjections] = this.adjustKernels(this.kernels, this.localPCs); %K_1(X_i, X_j) and S(X_i, X_j)
         weightedLinearSpacesProjections = cell(this.sampleSize);
         diagonalLinearSpaceProjections = cell(this.sampleSize,1);
         diagonalLinearSpaceProjectionsEigenValues = cell(this.sampleSize,1);
          eyeReducedDimension = eye(this.reducedDimension);
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
else
  phi2 = 0;
end
end

    
    newCompressedPoints = compress(model, newPoints);
    
    [decompressedPoints, failedPoints] = decompress(model, newPoints);
    
    plotPCs(model);
    
    plotProjectionJacobians(model);
    
  end
  
  methods (Hidden)
    
    kernels = calculateKernels(model, points, weights, otherPoints);
    
    [principalComponentsPerPoint, eigenValuesPerPoint] = calculateWeightedPCA(model, points, weights, otherPoints)
    
    [kernels, linearSpacesProjections] = adjustKernels(model, kernels, localPCs1, localPCs2)
    
    plotBases(model, bases);
    
  end
end