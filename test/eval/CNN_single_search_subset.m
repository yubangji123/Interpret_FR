% test caffe model on all folds
clear all;

load('../IJB-A/dataset.mat', 'croppedImages', 'sightingID2idx', 'badImage', 'poorIndex')
%load('CGAN/data/IJBA_Luan/PIFA_combine.mat', 'align_error')

N = numel(croppedImages);
poor_quality = find(poorIndex==1);
feature_blob = {'pool5'};

fileID = fopen('IJBA_features_iter_0.txt');
features_cell = textscan(fileID,'%f');
fclose(fileID);
IJBA_features = reshape(features_cell{1}, 320, [])';

features = IJBA_features;
features_mr = IJBA_features;

thr = 1;
good = ones(N, 1);
%good(IJBA_coefficients < thr) = 0;
%good(poor_quality) = 0;
result = struct();

setting = 'close'; % open or close-set face identification
for fd = 1 : 10
    tic
    
    gallery = load(['../IJB-A/search/split', num2str(fd), '/search_gallery_', num2str(fd), '.mat']);
    probe = load(['../IJB-A/search/split', num2str(fd), '/search_probe_', num2str(fd), '.mat']);
    
    idx = find(sum(gallery.visibility, 2)==3);
    good(sightingID2idx(gallery.sightingID(idx))) = 0;
    idx = find(sum(probe.visibility, 2)==3);
    good(sightingID2idx(probe.sightingID(idx))) = 0;
    
    uniqueIdx = unique(sightingID2idx([gallery.sightingID; probe.sightingID]));
    croppedImages1 = croppedImages(uniqueIdx);
     
%     fileID = fopen('CGAN/IJBA_features.txt');
%     gallery_features_cell = textscan(fileID,'%f');
%     features = reshape(gallery_features_cell{1}, 320, [])';
%     fclose(fileID);
%     
%     fileID = fopen('CGAN/IJBA_features_mirror.txt');
%     gallery_features_cell = textscan(fileID,'%f');
%     features_mr = reshape(gallery_features_cell{1}, 320, [])';
%     fclose(fileID);



    disp('feature extraction done!')
    
    gallery_temp = unique(gallery.templateID);
    nGallery = numel(gallery_temp);
    gallery_indexs = cell(nGallery, 1);
    gallery_subject = zeros(nGallery, 1);
    for p = 1 : nGallery
        idx = find(gallery.templateID == gallery_temp(p));
        sigh = gallery.sightingID(idx);
        valid = find(good(sightingID2idx(sigh))==1);
        if ~isempty(valid)
            sigh = sigh(valid);
            gallery_indexs{p} = sightingID2idx(sigh);
        end
        gallery_subject(p) = unique(gallery.subjectID(idx));
    end
    
    % image selection in each template for probe set
    prob_temp = unique(probe.templateID);
    nProbe = numel(prob_temp);
    probe_indexs = cell(nProbe, 1);
    probe_subject = zeros(nProbe, 1);
    closeset = zeros(nProbe, 1);
    for p = 1 : nProbe
        idx = find(probe.templateID == prob_temp(p));
        
        sigh = probe.sightingID(idx);
        valid = find(good(sightingID2idx(sigh))==1);
        if ~isempty(valid)
            sigh = sigh(valid);
            probe_indexs{p} = sightingID2idx(sigh);
        end
        probe_subject(p) = unique(probe.subjectID(idx));
        if intersect(probe_subject(p), unique(gallery_subject))
            closeset(p) = 1;
        end
    end
    if strcmp(setting, 'close')
        indexs = find(closeset==1);
        nProbe = numel(indexs);
        prob_temp = prob_temp(indexs);
        probe_indexs = probe_indexs(indexs);
        probe_subject = probe_subject(indexs);
    end
    
    dist = zeros(nProbe, nGallery);
    for p = 1 : nProbe
        p_idx = probe_indexs{p};
        if ~isempty(p_idx)
            
            p_feature = features(p_idx,:);
            p_feature_mr = features_mr(p_idx,:);
            
            
            for g = 1 : nGallery
                g_idx = gallery_indexs{g};
                if ~isempty(g_idx)
                    g_feature = features(g_idx,:);
                    g_feature_mr = features_mr(g_idx,:);
                    d1 = pdist2(p_feature, g_feature, 'cosine');
                    d2 = pdist2(p_feature, g_feature_mr, 'cosine');
                    d3 = pdist2(p_feature_mr, g_feature, 'cosine');
                    d4 = pdist2(p_feature_mr, g_feature_mr, 'cosine');
                    d = (d1 + d2 + d3 + d4) / 4;
                    dist(p, g) = mean([min(d), min(d')]);
                end
            end
        end
    end
    
    % close-set recognition
    % rank-1 identification rate
    idx_Good_prob=find(sum(dist, 2));
    dist(find(dist==0)) = 100;
    [~, idx] = min(dist, [], 2);
    esLabels = gallery_subject(idx);
    acc = (sum(esLabels(idx_Good_prob)==probe_subject(idx_Good_prob))) / numel(idx_Good_prob) ;
    
    % rank 1 to 10
    ranks = zeros(nProbe, 1);
    allRank = cell(nProbe, 1);
    for i = 1 : nProbe
        if sum(dist(i,:)-100)~=0
            label = probe_subject(i);
            d = dist(i,:);
            [~, id2] = sort(d);
            id1 = find(gallery_subject==label);
            ranks(i) = find(id2==id1);
        end
    end
    nRank = 10;
    rankAcc = zeros(nRank, 1);
    for i = 1 : nRank
        rankAcc(i) = (sum(ranks(idx_Good_prob)<=i)) / numel(idx_Good_prob);
    end  
    
    result.dist{fd} = dist;
    result.esLabels{fd} = esLabels;
    result.acc(fd) = acc;
    result.rank{fd} = rankAcc;
    disp(['folder: ', num2str(fd), ' , rank-1 = ', num2str(acc)])
    
    toc
end

ranks = cell2mat(result.rank);
rank1 = ranks(1,:);
rank5 = ranks(5,:);
rank10 = ranks(10,:);
rank1_mean = mean(rank1);
rank1_std = std(rank1);
rank5_mean = mean(rank5);
rank5_std = std(rank5);
rank10_mean = mean(rank10);
rank10_std = std(rank10);


disp(['rank-1 = ', num2str(rank1_mean*100), ' \pm ', num2str(rank1_std * 100), ' , rank-5 = ',  num2str(rank5_mean*100), ' \pm ', num2str(rank5_std*100)])






