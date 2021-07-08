clc,clear;

state = 'train';

folder = '/157Dataset/data-chen.dongwen/icme_2017/original_data/Eval';
sub_folder = '/HeadEyeImages/'
%exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/salicon17/ACSalNet4/ACSalNet4/s1'
%exp_sub_folder = '/test'
exp_folder = '/157Dataset/data-chen.dongwen/icme_2017/graduation_project'
exp_sub_folder = '/predictions_salmap'
outfolder = '/MCP_1024';

mkdir([exp_folder outfolder]);

imlist = dir([folder sub_folder]);

vfov = 90;
headmove_h = 0:45:315
headmove_v = -45:45:45

length(imlist)

parfor i = 3:length(imlist)
%for i = 3:length(imlist)
    im_filename = imlist(i).name;
    im_filename
    filenum = regexp(im_filename, 'P', 'split');
    filenum = regexp(cell2mat(filenum(2)), '\.', 'split');
    filenum = cell2mat(filenum(1));

    img = imread([exp_folder exp_sub_folder '/P' filenum '_11.png']);
    imw = size(img, 1);
    out_wid = imw*2;
    out_len = imw;

    sal_out = zeros(out_len, out_wid, 3);
    im_salgan_0 = zeros(imw,imw*2);

%    for hh = 1:length(headmove_h)
%        for hv = 1:length(headmove_v)
    parfor hh = 1:length(headmove_h)
        parfor hv = 1:length(headmove_v)
            % read image

			hv1 = hv

            img = double(imread([exp_folder exp_sub_folder '/P' filenum '_' num2str(hv1) num2str(hh) '.png']));
            sal_out(:,:,1) = img;
            sal_out(:,:,2) = img;
            sal_out(:,:,3) = img;
            out = equi2cubic(sal_out, imw, vfov, -headmove_v(hv1));
            im_salgan = cubic2equi(-headmove_h(hh),cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
            im_salgan = im_salgan(:,:,1);

            % imwrite(im_salgan, [exp_folder outfolder '/SHE' filenum '_' num2str(hv) num2str(hh) '.png']);

            im_salgan = double(im_salgan)+im_salgan_0;
            im_salgan_0 = im_salgan;
       end
    end
    im_salgan = im_salgan./(hh*hv1);
%    im_salgan = im_salgan./(length(headmove_h)*length(headmove_v));
    im_salgan = im_salgan/max(max(im_salgan));
    im_salgan = imresize(im_salgan, [out_len out_wid]);
    imwrite(im_salgan, [exp_folder outfolder '/SHE' filenum '.png']);
end
