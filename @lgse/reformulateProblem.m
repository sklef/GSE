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