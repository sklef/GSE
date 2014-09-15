function calculateJacobianComponent(this, index, dimensionIndex, kernels, iteration_number)
normalization_coefficient = sum(kernels(:, index)) - kernels(index, index);
projection_matrix = this.localPCs{index} * this.localPCs{index}';
history = this.projectionJacobians{index}(:,dimensionIndex);
for iteration = 1:iteration_number
    for i = 1:this.sampleSize
        this.projectionJacobians{i}(:, dimensionIndex) = this.projectionJacobians{i}(:, dimensionIndex) + projection_matrix * kernels(index, i) * this.projectionJacobians{i}(:, dimensionIndex) * normalization_coefficient; 
    end
    iteration
    history = [history, this.projectionJacobians{index}(:, dimensionIndex)];
    
end  
if dimensionIndex == 1
    this.history{index} = history;
end

end