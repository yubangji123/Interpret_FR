clear all;
clc;

feat = load('IJBC_features_iter_0.txt');

load('ijbc_temp_ID_face_indx.mat');

features = feat;
features_mr = feat;

n_probe = length(ijbc_temp_ID_face_indx.probe_temp_ID);
g1_probe = length(ijbc_temp_ID_face_indx.G1_temp_ID);
g2_probe = length(ijbc_temp_ID_face_indx.G2_temp_ID);
g_probe = g1_probe + g2_probe;

probe_subj_ID = ijbc_temp_ID_face_indx.probe_subj_ID;
G1_subj_ID = ijbc_temp_ID_face_indx.G1_subj_ID;
G2_subj_ID = ijbc_temp_ID_face_indx.G2_subj_ID;
G_subj_ID = [G1_subj_ID; G2_subj_ID];

probe_temp_ID = ijbc_temp_ID_face_indx.probe_temp_ID;
G1_temp_ID = ijbc_temp_ID_face_indx.G1_temp_ID;
G2_temp_ID = ijbc_temp_ID_face_indx.G2_temp_ID;
G_temp_ID = [G1_temp_ID; G2_temp_ID];

probe_face_indx = ijbc_temp_ID_face_indx.probe_face_indx;
G1_face_indx = ijbc_temp_ID_face_indx.G1_face_indx;
G2_face_indx = ijbc_temp_ID_face_indx.G2_face_indx;
G_face_indx = [G1_face_indx; G2_face_indx];

closeset = zeros(n_probe, 1);
for p = 1:n_probe
    if intersect(probe_subj_ID(p), unique(G1_subj_ID))
        closeset(p) = 1;
    end
    if intersect(probe_subj_ID(p), unique(G2_subj_ID))
        closeset(p) = 1;
    end
end

indexs = find(closeset==1);
n_probe = numel(indexs);
probe_temp_ID = probe_temp_ID(indexs);
probe_face_indx = probe_face_indx(indexs);
probe_subj_ID = probe_subj_ID(indexs);

dist = zeros(n_probe, g_probe);
for p = 1 : n_probe
    p_idx = probe_face_indx{p};
    if ~isempty(p_idx)
        
        p_feature = mean(features(p_idx,:), 1);
        
        for g = 1 : g_probe
            g_idx = G_face_indx{g};
            if ~isempty(g_idx)
                g_feature = mean(features(g_idx,:), 1);
                d = pdist2(p_feature, g_feature, 'cosine');
                dist(p, g) = mean([min(d), min(d')]);
            end
        end
    end
    if mod(p, 100) == 0
        display(p);
    end
end

% close-set recognition
% rank-1 identification rate
v_row = find(dist(:,1));
v_col = find(dist(1,:));
dist = dist(v_row, v_col);
G_subj_ID = G_subj_ID(v_col);
probe_subj_ID = probe_subj_ID(v_row);
n_probe = length(v_row);
closeset = zeros(n_probe, 1);
for p = 1:n_probe
    if intersect(probe_subj_ID(p), unique(G_subj_ID))
        closeset(p) = 1;
    end
end
indexs = find(closeset==1);
probe_subj_ID = probe_subj_ID(indexs);
dist = dist(indexs, :);
n_probe = length(probe_subj_ID);

[~, idx] = min(dist, [], 2);
esLabels = G_subj_ID(idx);
acc = (sum(esLabels==probe_subj_ID)) / n_probe ;

% rank 1 to 10
ranks = zeros(n_probe, 1);
allRank = cell(n_probe, 1);
for i = 1 : n_probe
    label = probe_subj_ID(i);
    d = dist(i,:);
    [~, id2] = sort(d);
    id1 = find(G_subj_ID==label);
    ranks(i) = find(id2==id1);
end
nRank = 10;
rankAcc = zeros(nRank, 1);
for i = 1 : nRank
    rankAcc(i) = (sum(ranks<=i)) / n_probe;
end


disp(['IJB-C: rank-1 = ', num2str(acc)])

