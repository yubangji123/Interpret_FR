clear all;
clc;

%% Preprocess all the images
% ijbc_metadata_path = 'protocols/ijbc_metadata.csv';
% all_data = readtable(ijbc_metadata_path);
% meta_subj_ID = all_data.SUBJECT_ID;
% meta_file_NA = all_data.FILENAME;
% 
% valid_idx = [];
% fid = fopen('IJBC_all_faces_path.txt', 'w+');
% for i = 1:length(meta_subj_ID)
%     img_name_s = strsplit(meta_file_NA{i}, '/');
%     img_name_s_s = strsplit(img_name_s{2}, '.');
%     img_path = sprintf(['/home/yinbangj/Bangjie/IJB/IJB/IJB-C/crops_output_new/', img_name_s{1}, ...
%         '/%d_', img_name_s_s{1}, '.jpg'], meta_subj_ID(i));
%     if exist(img_path)
%         valid_idx = [valid_idx; i];
%         fprintf(fid, [img_path, '\n']);
%     end
%     i
% end
% fclose(fid);

%% Preprocess the occluded images
ijbc_metadata_path = 'protocols/ijbc_metadata.csv';
all_data = readtable(ijbc_metadata_path);
meta_subj_ID = all_data.SUBJECT_ID;
meta_file_NA = all_data.FILENAME;
meta_file_OCC = [all_data.OCC1, all_data.OCC2, all_data.OCC3, all_data.OCC4, ...
    all_data.OCC5, all_data.OCC6, all_data.OCC7, all_data.OCC18,all_data.OCC9, ...
    all_data.OCC10, all_data.OCC11, all_data.OCC12, all_data.OCC13, all_data.OCC14, ...
    all_data.OCC15, all_data.OCC16, all_data.OCC17, all_data.OCC18];

locs_occl = [7, 8, 9];
locs_non_occl = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18];

valid_idx = [];
fid = fopen('IJBC_occluded_faces_path.txt', 'w+');
for i = 1:length(meta_subj_ID)
    img_name_s = strsplit(meta_file_NA{i}, '/');
    img_name_s_s = strsplit(img_name_s{2}, '.');
    img_path = sprintf(['/home/yinbangj/Bangjie/IJB/IJB/IJB-C/crops_output_new/', img_name_s{1}, ...
        '/%d_', img_name_s_s{1}, '.jpg'], meta_subj_ID(i));
    if exist(img_path)
        if sum(meta_file_OCC(i, :))> 0
            valid_idx = [valid_idx; i];
            fprintf(fid, [img_path, '\n']);
        end
    end
    i
end
fclose(fid);

%valid_idx = load('IJBC_features_idx.txt');
meta_subj_ID = meta_subj_ID(valid_idx);
for i = 1:length(valid_idx)
    meta_file_NA_tmp{i,1}  = meta_file_NA{valid_idx(i)};
end
meta_file_NA = meta_file_NA_tmp;


ijbc_1N_probe_mixed_path = 'protocols/ijbc_1N_probe_mixed.csv';
all_data = readtable(ijbc_1N_probe_mixed_path);
probe_temp_ID = all_data.TEMPLATE_ID;
probe_subj_ID = all_data.SUBJECT_ID;
probe_file_NA = all_data.FILENAME;
probe_temp_ID_single = unique(probe_temp_ID);
for i = 1:length(probe_temp_ID_single)
   p_tmp_id =  probe_temp_ID_single(i);
   p_tmp_id_indx = find(probe_temp_ID == p_tmp_id);
   p_subj_id = unique(probe_subj_ID(p_tmp_id_indx));
   m_subj_id = find(meta_subj_ID==p_subj_id);
   l = 1;
   face_indx = [];
   for j = 1:length(p_tmp_id_indx)
       for k = 1:length(m_subj_id)
           if isequal(probe_file_NA{p_tmp_id_indx(j)}, meta_file_NA{m_subj_id(k)})
               face_indx(l) = m_subj_id(k);
               l = l + 1;
               break;
           end
       end
   end
   probe_temp_ID_face_indx{i,1} = face_indx;
   probe_temp_subj_ID(i,1) = p_subj_id;
   if mod(i, 100) == 0
       i
   end
end

ijbc_1N_gallery_G1_path = 'protocols/ijbc_1N_gallery_G1.csv';
all_data = readtable(ijbc_1N_gallery_G1_path);
G1_temp_ID = all_data.TEMPLATE_ID;
G1_subj_ID = all_data.SUBJECT_ID;
G1_file_NA = all_data.FILENAME;
G1_temp_ID_single = unique(G1_temp_ID);
for i = 1:length(G1_temp_ID_single)
   g1_tmp_id =  G1_temp_ID_single(i);
   g1_tmp_id_indx = find(G1_temp_ID == g1_tmp_id);
   g1_subj_id = unique(G1_subj_ID(g1_tmp_id_indx));
   m_subj_id = find(meta_subj_ID==g1_subj_id);
   l = 1;
   face_indx = [];
   for j = 1:length(g1_tmp_id_indx)
       for k = 1:length(m_subj_id)
           if isequal(G1_file_NA{g1_tmp_id_indx(j)}, meta_file_NA{m_subj_id(k)})
               face_indx(l) = m_subj_id(k);
               l = l + 1;
               break;
           end
       end
   end
   G1_temp_ID_face_indx{i,1} = face_indx;
   G1_temp_subj_ID(i,1) = g1_subj_id;
   if mod(i, 100) == 0
       i
   end
end

ijbc_1N_gallery_G2_path = 'protocols/ijbc_1N_gallery_G2.csv';
all_data = readtable(ijbc_1N_gallery_G2_path);
G2_temp_ID = all_data.TEMPLATE_ID;
G2_subj_ID = all_data.SUBJECT_ID;
G2_file_NA = all_data.FILENAME;
G2_temp_ID_single = unique(G2_temp_ID);
for i = 1:length(G2_temp_ID_single)
   g2_tmp_id =  G2_temp_ID_single(i);
   g2_tmp_id_indx = find(G2_temp_ID == g2_tmp_id);
   g2_subj_id = unique(G2_subj_ID(g2_tmp_id_indx));
   m_subj_id = find(meta_subj_ID==g2_subj_id);
   l = 1;
   face_indx = [];
   for j = 1:length(g2_tmp_id_indx)
       for k = 1:length(m_subj_id)
           if isequal(G2_file_NA{g2_tmp_id_indx(j)}, meta_file_NA{m_subj_id(k)})
               face_indx(l) = m_subj_id(k);
               l = l + 1;
               break;
           end
       end
   end
   G2_temp_ID_face_indx{i,1} = face_indx;
   G2_temp_subj_ID(i,1) = g2_subj_id;
   if mod(i, 100) == 0
       i
   end
end
ijbc_temp_ID_face_indx.probe_temp_ID = probe_temp_ID_single;
ijbc_temp_ID_face_indx.probe_subj_ID = probe_temp_subj_ID;
ijbc_temp_ID_face_indx.probe_face_indx = probe_temp_ID_face_indx;
ijbc_temp_ID_face_indx.G1_temp_ID = G1_temp_ID_single;
ijbc_temp_ID_face_indx.G1_subj_ID = G1_temp_subj_ID;
ijbc_temp_ID_face_indx.G1_face_indx = G1_temp_ID_face_indx;
ijbc_temp_ID_face_indx.G2_temp_ID = G2_temp_ID_single;
ijbc_temp_ID_face_indx.G2_subj_ID = G2_temp_subj_ID;
ijbc_temp_ID_face_indx.G2_face_indx = G2_temp_ID_face_indx;

save 'ijbc_temp_ID_face_indx.mat' ijbc_temp_ID_face_indx

