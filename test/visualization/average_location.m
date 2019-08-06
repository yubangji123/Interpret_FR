%% Average locations of maximum feature response
load('tVM.mat');
load('landmarks68_IJBA.mat');
load('landmark68_7.mat');
IJBA_feature_name = {'IJBA_feature_maps_ours.txt','IJBA_feature_maps_spatial.txt',...
    'IJBA_feature_maps_ours.txt'};
type_name = {'base', 'spatial', 'ours'};
if ~exist(['avg_center_new_pos_base.mat'])
    for feat_i = 1:length(IJBA_feature_name)
        if exist(IJBA_feature_name{feat_i})
            y = load(IJBA_feature_name{feat_i});
            save([IJBA_feature_name{feat_i}(1:length(IJBA_feature_name{feat_i})-4), '.mat'], 'y');
            delete(IJBA_feature_name{feat_i});
        end
        load([IJBA_feature_name{feat_i}(1:length(IJBA_feature_name{feat_i})-4), '.mat']);
        k5 = y;
        s_num = size(k5, 1);
        f_dim = 320;
        res = 96;
        flag = {'pos', 'neg'};
        for flg_i = 1:length(flag)
            avg_center_new = zeros(f_dim, 2);
            front_id = 8;
            front_lmks = landmarks68{front_id};
            for j = 1:f_dim
                ts_vertice = zeros(s_num, 3);
                centers = zeros(s_num, 2);
                weights = zeros(s_num, 1);
                for i = 1:s_num
                    x = reshape(k5(i,:), f_dim, 24*24);
                    z = reshape(x(j,:), 24, 24);
                    z = z';
                    if isequal(flag{flg_i}, 'neg')
                        if min(z(:)) < 0
                            [row_p, col_p] = find(z == min(z(:)));
                        else
                            continue;
                        end
                    end
                    if isequal(flag{flg_i}, 'pos')
                        if max(z(:)) > 0
                            [row_p, col_p] = find(z == max(z(:)));
                        else
                            continue;
                        end
                    end
                    center_j = [row_p(1), col_p(1)] * 4;
                    t_num = size(tVM, 1);
                    for k = 1:t_num
                        t_vertice = find(tVM(k, :));

                        xl = landmarks68{i}(1, t_vertice);
                        yl = landmarks68{i}(2, t_vertice);

                        x = center_j(2);
                        y = center_j(1);

                        b1 = mySign([x,y], [xl(1), yl(1)], [xl(2), yl(2)]) < 0.0;
                        b2 = mySign([x,y], [xl(2), yl(2)], [xl(3), yl(3)]) < 0.0;
                        b3 = mySign([x,y], [xl(3), yl(3)], [xl(1), yl(1)]) < 0.0;



                        alpha(1,1) = ((x-xl(3))*(yl(2)-yl(3))-(y-yl(3))*(xl(2)-xl(3)))/...
                            ((xl(1)-xl(3))*(yl(2)-yl(3))-(yl(1)-yl(3))*(xl(2)-xl(3)));
                        alpha(2,1) = ((x-xl(3))*(yl(1)-yl(3))-(y-yl(3))*(xl(1)-xl(3)))/...
                            ((xl(2)-xl(3))*(yl(1)-yl(3))-(yl(2)-yl(3))*(xl(1)-xl(3)));
                        alpha(3,1) = 1 - alpha(1,1) - alpha(2,1);

                        if ( 0 <= alpha(1)) &&( alpha(1) <= 1) && ( 0 <= alpha(2)) && ( alpha(2)<= 1)  && ((alpha(1) + alpha(2)) <= 1)
                            ts_vertice(i, :) = t_vertice;
                        end
                    end

                    xl_tar = front_lmks(1, ts_vertice(i, :));
                    yl_tar = front_lmks(2, ts_vertice(i, :));

                    xl = landmarks68{i}(1, ts_vertice(i, :));
                    yl = landmarks68{i}(2, ts_vertice(i, :));

                    x = center_j(2);
                    y = center_j(1);

                    alpha(1,1) = ((x-xl(3))*(yl(2)-yl(3))-(y-yl(3))*(xl(2)-xl(3)))/...
                        ((xl(1)-xl(3))*(yl(2)-yl(3))-(yl(1)-yl(3))*(xl(2)-xl(3)));
                    alpha(2,1) = ((x-xl(3))*(yl(1)-yl(3))-(y-yl(3))*(xl(1)-xl(3)))/...
                        ((xl(2)-xl(3))*(yl(1)-yl(3))-(yl(2)-yl(3))*(xl(1)-xl(3)));
                    alpha(3,1) = 1 - alpha(1,1) - alpha(2,1);

                    x_new = xl_tar*alpha;
                    y_new = yl_tar*alpha;
                    if x_new<96 && y_new<96 && x_new>0 && y_new>0
                        centers(i, :) =  [y_new, x_new];
                        weights(i, 1) = abs(z(row_p(1), col_p(1)));
                    end
                end
                weights = weights/sum(weights);
                avg_center_new(j, :) = weights' * centers;
                devi_center_new(j, 1) = weights(find(weights))' * pdist2(centers(find(sum(centers, 2)), :), avg_center_new(j, :), 'euclidean');
                display(j);
            end

            save(['avg_center_new_', flag{flg_i},'_', type_name{feat_i},'.mat'], 'avg_center_new');
            save(['devi_center_new_', flag{flg_i},'_', type_name{feat_i},'.mat'], 'devi_center_new');
        end
    end
end
% Average location and standard deviation
t_name = {'base', 'spatial', 'ours'};
flag = {'pos', 'neg'};
ra = 1;
res = 96;
gap =10;
max_all = [];
min_all = [];
for i = 1:numel(t_name)
    for j = 1:numel(flag)
        load(['devi_center_new_', flag{j}, '_', t_name{i},'.mat']);
        max_all = [max_all, max(devi_center_new(:))];
        min_all = [min_all, min(devi_center_new(:))];
    end
end
max_devi = max(max_all(:));
min_devi = min(min_all(:));
for i = 1:numel(t_name)
    big_img = ones(res, res*2 + 1*gap, 3);
    for j = 1:numel(flag)
        load(['avg_center_new_', flag{j}, '_', t_name{i},'.mat']);
        load(['devi_center_new_', flag{j}, '_', t_name{i},'.mat']);
        img = imread('sample_image/img_7.png');
        im_c = zeros(size(img,1), size(img,2));
        for k = 1:size(avg_center_new, 1)
            x_n = avg_center_new(k, 1);
            y_n = avg_center_new(k, 2);
            im_c(floor(x_n), floor(y_n)) = devi_center_new(k);
        end
        img = double(img)/255;
        im_s = showWithColorMap(im_c, [min_devi, max_devi]);
        for a = 1:size(im_s,1)
            for b = 1:size(im_s,2)
                if im_s(a,b,1) ~= 0 || im_s(a,b,2) ~= 0 || im_s(a,b,3)>im_s(1,1,3)
                    img(a,b,:) = im_s(a,b,:);
                end
            end
        end
        big_img(:, (j-1)*res+1+(j-1)*gap:j*res+(j-1)*gap, :) = img;
    end
    figure;
    imshow(big_img);
    colormap jet;
    c = colorbar;
    c.FontSize = 30;
    caxis([min(im_c(:)) max(im_c(:))]);
    %saveas(gcf, ['/home/yinbangj/Bangjie/rebuttal/color_avg_devi_', flag{j}, '_', t_name{i}, '_', num2str(ra),'.png']);
end