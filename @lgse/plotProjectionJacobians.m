function plotProjectionJacobians(this)

this.plotBases(this.projectionJacobians)
return

starts = [];
ends = [];

startsOrth = this.trainPoints;
endsOrth = [];


starts = this.trainPoints;
ends1 = [];
ends2 = [];
for pointIndex = 1:this.sampleSize
%   starts = [starts; this.points(pointIndex,:); this.points(pointIndex,:)];
%   ends = [ends; this.projectionJacobians{pointIndex}'];
  ends1 = [ends1; this.projectionJacobians{pointIndex}(:,1)'];
  ends2 = [ends2; this.projectionJacobians{pointIndex}(:,2)'];
%   startsOrth = [startsOrth; this.points(pointIndex,:);];
  M = this.projectionJacobians{pointIndex}*(this.projectionJacobians{pointIndex}' * this.projectionJacobians{pointIndex})^0.5;
  [t, ~] = eig(M * M');
  endsOrth = [endsOrth; t(:, 1)'];
end
% ends = ends * 0.1;
% endsOrth = endsOrth * 0.1;

quiver3(starts(:,1), starts(:,2), starts(:,3), ends1(:,1), ends1(:,2), ends1(:,3), 0)
hold on
quiver3(starts(:,1), starts(:,2), starts(:,3), ends2(:,1), ends2(:,2), ends2(:,3), 0)
% quiver3(startsOrth(:,1), startsOrth(:,2), startsOrth(:,3), endsOrth(:,1), endsOrth(:,2), endsOrth(:,3), 'r')
hold off

end