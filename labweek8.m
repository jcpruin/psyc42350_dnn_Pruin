%% set up matconvnet

addpath('matconvnet-master')

run matconvnet-master/matlab/vl_setupnn;

net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

%% Make the Image the correct size for the package to use

avg_im = net.meta.normalization.averageImage;
avg_im = (avg_im - min(min(min(avg_im))))/max(max(max(avg_im)));
imshow(avg_im)

%% connect to RSA stim
addpath('matconvnet-master/RSA stimuli')

imnames = {'cave.jpg', 'icecave.jpg', 'forestcanopy.jpg', 'garden.jpg', 'lake.jpg', 'mountains.jpg', 'bathroom.jpg', 'conferencehall.jpg', 'livingroom.jpg', 'city.jpg', 'farmhouse.jpg', 'suburbs.jpg'};

%create proper frame size and apply function to RSA stim
cnnoutputs = cell(1, length(imnames));
for im = 1:length(imnames)
    cnnoutputs{im} = metanorm(imnames{im});
end

%% create RDM of RSA stim
net_rdms = cell(1, 8); 

for lay = 1:8
net_rdm = zeros(length(imnames), length(imnames));
    for im1 = 1:length(imnames)
        for im2 = 1:length(imnames)
            net_rdm(im1,im2) = 1 - corr(cnnoutputs{im1}{lay}', cnnoutputs{im2}{lay}');
        end
    end
    net_rdms{lay} = net_rdm;  
end

%% Make a pretty picture of the RDM
for lay = 1:8
    subplot(2, 4, lay)
    imshow(net_rdms{lay}, 'InitialMagnification','fit')
end

%% use Week 4 function and code to compare visual RDM and category RDM performance

flatnn = extractRDM(net_rdms{8});
flatvis = extractRDM(visualrdm); 
flatcat = extractRDM(categoryrdm);

[rho1,p1] = corr(flatnn, flatvis, 'type', 'Spearman');
[rho2,p2] = corr(flatnn, flatcat, 'type', 'Spearman');


