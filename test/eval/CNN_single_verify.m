% test caffe model on all folds

% currentpath = '/home/yinxi/Documents/MATLAB/face_recognition/IJB-A';
% addpath('/home/yinxi/Documents/MATLAB/face_recognition/');
% caffepath = '/home/yinxi/Documents/git_caffe/';
% cd(currentpath)

data = load('IJB-A/IJBA_crop.mat');

load('IJB-A/dataset.mat', 'sightingID2idx', 'badImage', 'poorIndex')
%load('CGAN/data/IJBA_Luan/PIFA_combine.mat', 'align_error')
 croppedImages=data.croppedImages;
N = numel(data.croppedImages);
poor_quality = find(poorIndex==1);
feature_blob = {'pool5'};

fileID = fopen('IJBA_features_iter_0.txt');
features_cell = textscan(fileID,'%f');
fclose(fileID);
IJBA_features = reshape(features_cell{1}, 320, [])';


features = IJBA_features;
features_mr = IJBA_features;

thr = 0;
good = ones(N, 1);
%good(IJBA_coefficients < thr) = 0;
%good(poor_quality) = 0;
result = struct();

total = 0;
selected  = 0;
fail_idx = 1;
img_ids = [];

for fd = 1 : 10
%     fileID = fopen(sprintf('CGAN/IJBA_features_iter_0_fold_%d.txt', fd));
% features_cell = textscan(fileID,'%f');
% fclose(fileID);
% IJBA_features2 = reshape(features_cell{1}, 320, [])';
    tic
    
    load(['IJB-A/verify/split', num2str(fd), '/verify_comparisons_', num2str(fd), '.mat'])
    load(['IJB-A/verify/split', num2str(fd), '/verify_metadata_', num2str(fd), '.mat'])
    uniqueIdx = unique(sightingID2idx(sightingID));
    croppedImages1 = croppedImages(uniqueIdx);
     
    disp('feature extraction done!')
    
    % image selection for each template
    templates = unique(templateID);
    nTemp = numel(templates);
    temp_indexs = cell(nTemp, 1);
    temp_subject = zeros(nTemp, 1);
    count = 0;
    for p = 1 : nTemp
        idx = find(templateID == templates(p));
        if length(idx) == 1
            temp_indexs{p} = sightingID2idx(sightingID(idx));
        else
            sigh = sightingID(idx);            total = total + numel(sigh);

            valid = find(good(sightingID2idx(sigh))==1);
            if isempty(valid)
                d = IJBA_coefficients(sightingID2idx(sigh));
                [~, valid] = max(d);
                
                count = count + 1;
            end
            sigh = sigh(valid);
            
            selected = selected + numel(valid);
            temp_indexs{p} = sightingID2idx(sigh);
        end
        temp_subject(p) = unique(subjectID(idx));
    end
    
    disp(count);
    
    % set-to-set face verification
    nCom = length(comparison);
    dist = zeros(nCom, 3);
    same = zeros(nCom, 1);
    for c = 1 : nCom
        temp1 = comparison(c, 1);
        temp2 = comparison(c, 2);
        id1 = find(templates==temp1);
        id2 = find(templates==temp2);
       
         feature1 = features(temp_indexs{id1},:);
         feature2 = features(temp_indexs{id2},:);
         feature1_mr = features_mr(temp_indexs{id1},:);
         feature2_mr = features_mr(temp_indexs{id2},:);
         
         d1 = pdist2(feature1, feature2, 'cosine');
         d2 = pdist2(feature1, feature2_mr, 'cosine');
         d3 = pdist2(feature1_mr, feature2, 'cosine');
         d4 = pdist2(feature1_mr, feature2_mr, 'cosine');
         
         
         d = (d1 + d2 + d3 + d4) / 4;

        
        
        dist(c, 1) = mean(d(:)); 
        dist(c, 2) = min(d(:)); 
        dist(c, 3) = mean([min(d), min(d')]); 

        same(c) = temp_subject(id1)==temp_subject(id2);
        %if mod(c, 1000) == 0
        %    disp(c)
        %end
    end

    tar = zeros(nCom, 3);
    far = zeros(nCom, 3);
    acc = zeros(3, 2); % for three metrics @0.01 and @0.001
    for i = 1 : 3
        [d0, idx] = sort(dist(:,i));
        same0 = same(idx);
        for j = 1 : nCom
            tar(j, i) = sum(same0(1:j)==1) / sum(same0==1);
            far(j, i) = sum(same0(1:j)==0) / sum(same0==0);
        end
        [~, id1] = min(abs(far(:,i)-0.01));
        [~, id2] = min(abs(far(:,i)-0.001));
        acc(i, 1) = tar(id1, i);
        acc(i, 2) = tar(id2, i);
    end
    result.dist{fd} = dist;
    result.same{fd} = same;
    result.tar{fd} = tar;
    result.far{fd} = far;
    result.acc{fd} = acc;
    
    %real = comparison(idx(find(same0(1:length(find(same==1)))==0)), :);
%     real = comparison(idx(find(same0((length(find(same==1))+1):end)==1) + length(find(same==1))), :);
%     for ii = 1:size(real, 1)
%         id1 = find(templates==real(ii, 1));
%         id2 = find(templates==real(ii, 2));
%         img_ids(fail_idx,1) = temp_indexs{id1}(1);
%         img_ids(fail_idx,2) = temp_indexs{id2}(1);
%         fail_idx = fail_idx + 1;
%     end
%     save('img_ids.mat', 'img_ids');
    
    disp(['folder: ', num2str(fd), ' , acc@0.01FAR = ', num2str(acc(3,1)), ' , acc@0.001FAR = ', num2str(acc(3,2))])
    toc
end


disp(selected/total);
acc = cell2mat(result.acc);
acc1 = acc(3, 1:2:end);
acc2 = acc(3, 2:2:end);
acc1_mean = mean(acc1);
acc1_std = std(acc1);
acc2_mean = mean(acc2);
acc2_std = std(acc2);

disp(['acc@0.01FAR = ', num2str(acc1_mean*100), ' \pm ', num2str(acc1_std * 100), ' , acc@0.001FAR = ',  num2str(acc2_mean*100), ' \pm ', num2str(acc2_std*100)])

