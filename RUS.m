%% This script performs Random UnderSampling on most populated classes

N = 208;
Dest = 'D:\Dev-Set-Rus\Truck';
Folder = 'D:\Dev-Set\Truck';
FileList = dir(fullfile(Folder,'*.wav'));
Index = randperm(numel(FileList),N);

for k = 1:N
  Source = fullfile(Folder, FileList(Index(k)).name);
  copyfile(Source, Dest);
end

