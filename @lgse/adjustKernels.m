function [kernels, linearSpacesProjections] = adjustKernels(this, kernels, localPCs1, localPCs2)
if nargin < 4
  localPCs2 = localPCs1;
end

linearSpacesProjections = cell(this.sampleSize);
for pointIndex1 = 1:size(kernels, 1)
  for pointIndex2 = 1:size(kernels, 2)
    linearSpacesProjections{pointIndex1, pointIndex2} = localPCs1{pointIndex1}'*localPCs2{pointIndex2}; %S(X_i, X_j) = Q_i^T Q_j
    linearSpacesProjectionsDeterminant = det(linearSpacesProjections{pointIndex1, pointIndex2});
    cauchyBinetDistance = (abs(1 - linearSpacesProjectionsDeterminant^2))^0.5;
    if cauchyBinetDistance < this.CauchyBinetMetricsThreshold
      kernels(pointIndex1, pointIndex2) = kernels(pointIndex1, pointIndex2)*linearSpacesProjectionsDeterminant^2;
    else
      kernels(pointIndex1, pointIndex2) = 0;
    end
    
  end
end
end