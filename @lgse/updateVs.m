function updateVs(this, dimension)
  for pointIndex = 1:this.sampleSize
    this.vs{pointIndex}(:, dimension) = zeros(this.reducedDimension, 1);
    this.vs{pointIndex}(dimension, dimension) = norm(this.projectionJacobians{pointIndex}(:, dimension));
  end
end