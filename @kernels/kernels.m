classdef kernels < handle
    
    properties (SetAccess = private)
       values 
    end
    
    methods
        
        function this = kernels(type, pointsNew, pointsOld, width, maxDist)
            switch lower(type)
                case {'gaussian', 'gaus'}
                    euclideanDist = dist(pointsNew, pointsOld');
                    this.values = (euclideanDist < maxDist) .* exp(-width * euclideanDist);
            end
        end
        
        function bases = calculateTangentBases(this, type, points, dim)
            bases = cell(size(points, 1), 1);
            switch lower(type)
                case {'weightedpca', 'wpca'}
                    for pointIdx = 1:size(points, 1)
                        nonZeroIdx = this.values(pointIdx, :) > 0;
                        bases{pointIdx} = pca(points(nonZeroIdx, :), 'NumComponents', dim, 'Weights', this.values(nonZeroIdx));
                    end
                case {'pca'}
                    for pointIdx = 1:size(points, 1)
                        nonZeroIdx = this.values(pointIdx, :) > 0;
                        bases{pointIdx} = pca(points(nonZeroIdx, :), 'NumComponents', dim);
                    end
            end
        end
        
        
        function output = adjust(this, type, points, dim)
            switch lower(type)
                case {'binetcauchy', 'bc'}
                    Qs = cell(size(points, 1), 1);
                    for pointIdx = 1:size(points, 1)
                        nonZeroIdx = this.values(pointIdx, :) > 0;
                        Qs{pointIdx} = pca(points(nonZeroIdx, :), 'NumComponents', dim, 'Weights', this.values(nonZeroIdx));
                    end
            end
        end
    end
end