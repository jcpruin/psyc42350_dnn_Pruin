function cnnoutputs = metanorm(image)
%setup matconvnet
%most of this was copied and adapted from the matconvnet website
addpath('matconvnet-master/matlab');

run matconvnet-master/matlab/vl_setupnn;

net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

%obtain and preprocess image
im = imread(image);
im_ = single(im); % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

% Run the CNN.
res = vl_simplenn(net, im_);

% Show the classification result.
scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);
figure(1) ; clf ; imagesc(im);
title(sprintf('%s (%d), score %.3f',...
   net.meta.classes.description{best}, best, bestScore));

layers = [1 5 9 11 13 16 18 20]

%This part was modified to serve different purposes and I didn't save the
%in-between parts
cnnoutputs = cell(1, numel(layers));
for lay = 1:numel(layers)
    act = squeeze(gather(res(lay).x));
    act = reshape(act, [1, size(act, 1)*size(act, 2)*size(act, 3)]); 
    cnnoutputs{lay} = act;
end

