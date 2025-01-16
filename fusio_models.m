% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';

fixedSize = [128 128];

numBinsColorKnn = 16;

numBinsHOGKnn = 16;
cellSizeKnn = 32;
    
veinsLBPKnn = 8;
radiLBPKnn = 1;
veins = 1;

%% Carregar dades d'entrenament
trainData = [];
trainLabels = [];
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));

classNames = {trainSubfolders.name};

for i = 1:length(trainSubfolders)
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));
    
    for j = 1:length(imageFiles)
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);
        
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end

        img = imresize(img, fixedSize);

        hsvImg = rgb2hsv(img);
        
        histH = imhist(hsvImg(:,:,1), numBinsColorKnn);
        histS = imhist(hsvImg(:,:,2), numBinsColorKnn);
        histV = imhist(hsvImg(:,:,3), numBinsColorKnn);
        
        histH = histH / sum(histH);
        histS = histS / sum(histS);
        histV = histV / sum(histV);
        
        colorHistoFeatures = [histH; histS; histV]';
        grayImg = rgb2gray(img);
        lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', veinsLBPKnn, 'Radius', radiLBPKnn);
        [hogFeatures, visualization] = extractHOGFeatures(grayImg, 'CellSize', [cellSizeKnn cellSizeKnn], 'NumBins', numBinsHOGKnn);

        % combinedFeatures = [colorHistoFeatures, lbpFeatures, hogFeatures];
        %combinedFeatures = [colorHistoFeatures, hogFeatures];
        combinedFeatures = [colorHistoFeatures, lbpFeatures];
        
        trainData = [trainData; combinedFeatures];
        trainLabels = [trainLabels; i];
    end
    disp(['Finished processing ', trainSubfolders(i).name]);
end

%% Carregar dades de test
testData = [];
testLabels = [];
testSubfolders = dir(testFolder);
testSubfolders = testSubfolders([testSubfolders.isdir] & ~ismember({testSubfolders.name}, {'.', '..'}));

for i = 1:length(testSubfolders)
    folderPath = fullfile(testFolder, testSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));

    for j = 1:length(imageFiles)
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);
        
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end

        img = imresize(img, fixedSize);

        grayImg = rgb2gray(img);
        hsvImg = rgb2hsv(img);
        
        histH = imhist(hsvImg(:,:,1), numBinsColorKnn);
        histS = imhist(hsvImg(:,:,2), numBinsColorKnn);
        histV = imhist(hsvImg(:,:,3), numBinsColorKnn);
        
        histH = histH / sum(histH);
        histS = histS / sum(histS);
        histV = histV / sum(histV);
        
        colorHistoFeatures = [histH; histS; histV]';
        lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', veinsLBPKnn, 'Radius', radiLBPKnn);
        [hogFeatures, visualization] = extractHOGFeatures(grayImg, 'CellSize', [cellSizeKnn cellSizeKnn], 'NumBins', numBinsHOGKnn);

        %combinedFeatures = [colorHistoFeatures, lbpFeatures, hogFeatures];
        %combinedFeatures = [colorHistoFeatures, hogFeatures];
        combinedFeatures = [colorHistoFeatures, lbpFeatures];
        
        testData = [testData; combinedFeatures];
        testLabels = [testLabels; i];
    end
    disp(['Finished processing ', testSubfolders(i).name]);
end

%% KNN
Mdl = fitcknn(trainData, trainLabels, 'NumNeighbors', 1);

predictedLabels = predict(Mdl, testData);

accuracy = sum(predictedLabels == testLabels) / length(testLabels);
disp(['Accuracy: ', num2str(accuracy)]);

confMat = confusionmat(testLabels, predictedLabels);
figure;
confusionchart(confMat, classNames);
title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');

%% Feature Importance
RFModel = TreeBagger(100, trainData, trainLabels, 'OOBPredictorImportance', 'on');
importance = RFModel.OOBPermutedPredictorDeltaError;

figure;
bar(importance);
title('Feature Importance Estimates');
xlabel('Feature Index');
ylabel('Importance');

%%
numTrees = 100;
treeBaggerModel = TreeBagger(numTrees, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');

% Evaluar model
predictedLabels = str2double(predict(treeBaggerModel, testData));
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del modelo: %.2f%%\n', accuracy * 100);

%% Mostrar evolució del nombre d'arbres
% figure;
% oobError = oobError(mdl);
% plot(oobError);
% xlabel('Número darbres');
% ylabel('Error');
% title('Evoluci del error OOB');

%% Matriu de confusió
figure;
confMat = confusionmat(testLabels, predictedLabels);
confusionchart(confMat, classNames);
title('Matriu de confusió');

%%
% Guardar el modelo TreeBagger en un archivo
modelFileName = 'treeBaggerModel.mat';
save(modelFileName, 'treeBaggerModel');
disp(['Model saved to ', modelFileName]);
