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
    history
    localPCs
    localEigenVals
    
    currentDelta % current delta for optimization debug 
    historyDelta % current delta for optimization debug  
    historyDeltaCompression % current delta compression for optimization debug  
    
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
    
     calculateJacobianComponent(this, dimensionIndex, kernels, iteration_number)
     calculateDelta(this, kernels, dimensionIndex)
     calculateDeltaCompression(this, kernels)
     updatePCs(this, dimension)
     updateVs(this, dimension)
    
    constructCompressedSpace(model, kernels, iteration_number);
    
    train(model, points, reducedDimension);
     
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