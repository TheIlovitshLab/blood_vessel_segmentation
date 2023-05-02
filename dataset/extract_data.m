clear; clc;

% all_ID = {'RG6 - MB 150', 'RG11 - MB 175'};
raw_data_dir = dir("raw_data");
all_ID = {raw_data_dir.name};
all_ID = all_ID(:, 3:end);

for n = 1:length(all_ID)
    ID = all_ID{n};
    verified = load(fullfile('raw_data', ID, 'User Verified Table.mat')).userVerified;
    
    for k = 1:size(verified,1)
        if verified{k, 2} == 1
            img_name = verified{k, 1};

            % load mask
            base_name = strsplit(img_name, '.');
            mask_file = load(fullfile('raw_data', ID, strcat(base_name{1}, '.mat')));
            mask = mask_file.derivedPic.BW_2;
    
            % save img and mask to train dir
            copyfile(fullfile('raw_data', ID, img_name), fullfile('train','images',img_name));
            
            save_maks_name = strcat(base_name{1}, '.mat');
            save_mask_dir = fullfile('train', 'masks', save_maks_name);
            save(save_mask_dir, 'mask')
%             imwrite(mask, save_mask_dir)
            
        end
    end
end
