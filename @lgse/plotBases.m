function plotBases(this, bases)
starts = [];
ends = [];

startsOrth = [];
endsOrth = [];

for pointIndex = 1:this.sampleSize
  starts = [starts; this.trainPoints(pointIndex,:); this.trainPoints(pointIndex,:)];
  ends = [ends; bases{pointIndex}'];
  startsOrth = [startsOrth; this.trainPoints(pointIndex,:);];
  M = orth(bases{pointIndex});%*(bases{pointIndex}' * bases{pointIndex})^0.5;
  [t, ~] = eig(M * M');
  endsOrth = [endsOrth; t(:, 1)'];
end
colors = {'b', 'm'};
for dimension = 2 % this.reducedDimension % 1:this.reducedDimension
  quiver3(starts(2 * (0:this.sampleSize-1) + dimension,1), ...
          starts(2 * (0:this.sampleSize-1) + dimension,2), ...
          starts(2 * (0:this.sampleSize-1) + dimension,3), ...
          ends(2 * (0:this.sampleSize-1) + dimension,1), ...
          ends(2 * (0:this.sampleSize-1) + dimension,2), ...
          ends(2 * (0:this.sampleSize-1) + dimension,3), ...
          colors{dimension});
  hold on
end
%quiver3(startsOrth(:,1), startsOrth(:,2), startsOrth(:,3), endsOrth(:,1), endsOrth(:,2), endsOrth(:,3), 'r')
hold off

end