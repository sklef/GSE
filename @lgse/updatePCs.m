function updatePCs(this, dimension)
  for pointIndex = 1:this.sampleSize
    currentH = this.projectionJacobians{pointIndex}(:, dimension);
    normCurrentH = norm(currentH);
    if normCurrentH > 0
      this.localPCs{pointIndex}(:, dimension) = currentH / normCurrentH;
    end
    this.localPCs{pointIndex}(:, dimension+1:end) = ...
      null([this.localPCs{pointIndex}(:, 1:dimension), null(this.localPCs{pointIndex}')]');
  end
end