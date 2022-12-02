my_files = dir(fullfile('C:\Users\lilac\Desktop\EB_reaver\segmentation model/raw_data','*.mat'));

for k = 1:length(my_files)
  base_file_name = my_files(k).name;
  full_file_name = fullfile('raw_data', base_file_name);
  file = load(full_file_name);
  map = file.derivedPic.wire;
  x = strsplit(base_file_name, '.');
  file_name = strcat(x{1}, '.png');
%   writematrix(map, file_name)
  imwrite(map, file_name)
end