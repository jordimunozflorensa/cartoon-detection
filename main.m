if exist('treeBaggerModel.mat', 'file')
    data = load('treeBaggerModel.mat');
    treeBaggerModel = data.treeBaggerModel;
else
    error('The file treeBaggerModel.mat does not exist.');
end

img = imread('DATA/train/Gumball/Gunball1.jpg');

fixedSize = [128 128];
numBins = 16;
veinsLBP = 8;
radiLBP = 1;

if size(img, 3) ~= 3
    img = cat(3, img, img, img);
end

img = imresize(img, fixedSize);

grayImg = rgb2gray(img);
hsvImg = rgb2hsv(img);

histH = imhist(hsvImg(:,:,1), numBins);
histS = imhist(hsvImg(:,:,2), numBins);
histV = imhist(hsvImg(:,:,3), numBins);

histH = histH / sum(histH);
histS = histS / sum(histS);
histV = histV / sum(histV);

colorHistoFeatures = [histH; histS; histV]';
lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', veinsLBP, 'Radius', radiLBP);

imgFeatures = [colorHistoFeatures, lbpFeatures];

[label, scores] = predict(treeBaggerModel, imgFeatures);

trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));
classNames = {trainSubfolders.name};

idx = label{1};
intidx = str2num(idx);
disp(['La imatge correspon a la sèrie: ', classNames{intidx}]);