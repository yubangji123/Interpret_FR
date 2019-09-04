clear all;
clc;

% if exist('IJBC_features.txt')
%feat = load('IJBC_features_iter_0.txt');
%     save 'IJBC_features.mat' feat;
%     delete('IJBC_features.txt');
% end
load('ijbc_temp_ID_face_indx.mat');
load('IJBC_features.mat');

ijbc_11_G1_G2_matches_path = 'protocols/ijbc_11_G1_G2_matches.csv';
all_data = csvread(ijbc_11_G1_G2_matches_path);

features = feat;
features_mr = feat;
n_com = size(all_data, 1);
step = 100000;
dist = [];
same = [];
for j = 1:ceil(n_com/step)
    tic
    valid_n_com = 1;
    dist_s = [];
    same_s = [];
    j_start = step*(j-1) + 1;
    j_end = step*j;
    if j_end > n_com
        j_end = n_com;
    end
    for i = j_start:j_end
        pair = all_data(i, :);
        temp1 = pair(1);
        temp2 = pair(2);
        
        id1_g1 = find(ijbc_temp_ID_face_indx.G1_temp_ID==temp1);
        id1_g2 = find(ijbc_temp_ID_face_indx.G2_temp_ID==temp1);
        if ~isempty(id1_g1)
            id1 = id1_g1;
            face_indx_1 = ijbc_temp_ID_face_indx.G1_face_indx{id1};
            temp_subject_1 = ijbc_temp_ID_face_indx.G1_subj_ID(id1);
        end
        if ~isempty(id1_g2)
            id1 = id1_g2;
            face_indx_1 = ijbc_temp_ID_face_indx.G2_face_indx{id1};
            temp_subject_1 = ijbc_temp_ID_face_indx.G2_subj_ID(id1);
        end
        id2 = find(ijbc_temp_ID_face_indx.probe_temp_ID==temp2);
        face_indx_2 = ijbc_temp_ID_face_indx.probe_face_indx{id2};
        temp_subject_2 = ijbc_temp_ID_face_indx.probe_subj_ID(id2);
        
        if ~isempty(face_indx_1) && ~isempty(face_indx_2)
            
            feature1 = mean(features(face_indx_1,:), 1);
            feature2 = mean(features(face_indx_2,:), 1);
            
            d = pdist2(feature1, feature2, 'cosine');
            
            dist_s(valid_n_com, 1) = mean(d(:));
            dist_s(valid_n_com, 2) = min(d(:));
            dist_s(valid_n_com, 3) = mean([min(d), min(d')]);
            
            same_s(valid_n_com) = temp_subject_1==temp_subject_2;
            
            valid_n_com = valid_n_com + 1;
        end
    end
    dist = [dist; dist_s];
    same = [same, same_s];
    toc
end
valid_n_com = numel(same);
tar = zeros(valid_n_com, 3);
far = zeros(valid_n_com, 3);
acc = zeros(3, 2); % for three metrics @0.01 and @0.001
for i = 1 : 3
    [d0, idx] = sort(dist(:,i));
    same0 = same(idx);
    
    same0_sum1 = sum(same0==1);
    same0_sum0 = sum(same0==0);
    same0_cul_sum1 = zeros(valid_n_com, 1);
    same0_cul_sum0 = zeros(valid_n_com, 1);
    same0_cul_sum1(1) = same0(1);
    same0_cul_sum0(1) = 1-same0(1);
    for j = 2 : valid_n_com
        if same0(j) == 1
            same0_cul_sum1(j) = same0_cul_sum1(j-1)+1;
            same0_cul_sum0(j) = same0_cul_sum0(j-1);
        else
            same0_cul_sum1(j) = same0_cul_sum1(j-1);
            same0_cul_sum0(j) = same0_cul_sum0(j-1)+1;
        end
        if mod(j, 10000) == 0
            display(j);
        end
    end
    for j = 1 : valid_n_com
        tar(j, i) = same0_cul_sum1(j) / same0_sum1;
        far(j, i) = same0_cul_sum0(j) / same0_sum0;
        if mod(j, 10000) == 0
            display(j);
        end
    end
    [~, id1] = min(abs(far(:,i)-0.01));
    [~, id2] = min(abs(far(:,i)-0.001));
    acc(i, 1) = tar(id1, i);
    acc(i, 2) = tar(id2, i);
end

disp(['IJB-C: acc@0.01FAR = ', num2str(acc(3,1)), ' , acc@0.001FAR = ', num2str(acc(3,2))])




