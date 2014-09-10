function constructCompressedSpace(this)
LHS = cell(this.sampleSize);
RHS = cell(this.sampleSize, 1);

for pointIndex1 = 1:this.sampleSize
  RHStmp = zeros(this.reducedDimension, 1);
  LHSdiag = zeros(this.reducedDimension);
  for pointIndex2 = [1:pointIndex1-1 pointIndex1+1:this.sampleSize]
    tmp = this.kernels(pointIndex1, pointIndex2) * (this.projectionJacobians{pointIndex1}'*this.projectionJacobians{pointIndex1} + this.projectionJacobians{pointIndex2}'*this.projectionJacobians{pointIndex2});
    LHS{pointIndex1, pointIndex2} = tmp;
    LHSdiag = LHSdiag - tmp;
    RHStmp = RHStmp + this.kernels(pointIndex1, pointIndex2) * ...
      ((this.projectionJacobians{pointIndex1}' + this.projectionJacobians{pointIndex2}') * ...
      (this.points(pointIndex2,:) - this.points(pointIndex1,:))');
  end
  LHS{pointIndex1, pointIndex1} = LHSdiag;
  RHS{pointIndex1} = RHStmp;
end

this.compressedPoints = reshape([cell2mat(LHS); repmat(eye(this.reducedDimension), 1, this.sampleSize)] \ ...
  [cell2mat(RHS); zeros(this.reducedDimension, 1)], this.reducedDimension, this.sampleSize);
end