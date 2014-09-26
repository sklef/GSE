load('MSE.mat');
surfaceNames = {'saddle', 'ellipsoid',  'cylinder'};
methods = {'GSE', 'OGSE'};
trainSizes = [250, 500, 1000, 2000];
styles = {'-vr', '-sb'};
for surfaceIndex = 1:length(surfaceNames)
  handle = figure();  
  for methodIndex = 1:length(methods)
    loglog(trainSizes, MSE(methodIndex, :, surfaceIndex), styles{methodIndex}, 'MarkerSize', 7, 'LineWidth', 3); % 
    hold on
  end  
  grid on
  xlim([125, 4000]);
  set(gca, 'XTick', [125, trainSizes, 4000]);
  legend(methods{:});
  xlabel('sample size');
  ylabel('MSE');
  title(surfaceNames{surfaceIndex});
  saveas(handle, strcat(surfaceNames{surfaceIndex}, 'MSE.png'));
end