% startup script to make Matlab aware of the MACROS GP package

disp(['executing MACROS GP startup script...']);

fileName = mfilename('fullpath');
sepIndices = find( fileName == filesep );
folderName = fileName(1:sepIndices(end));

allSubFolders = genpath(folderName);
folderSep = find( allSubFolders == pathsep );

ind2Del = false( size(allSubFolders) );

for i = 2 : length(folderSep)
    if any ( strfind( allSubFolders( folderSep(i - 1) + 1 :  folderSep(i) - 1 ), '.svn') )
        ind2Del( folderSep(i - 1) + 1 :  folderSep(i) ) = true;
    end
end

%delete
allSubFolders(ind2Del) = [];
addpath(allSubFolders); 

clear fileName folderName allSubFolders ind2Del folderSep i sepIndices