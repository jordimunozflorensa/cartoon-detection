% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';
% Numero de bins
numBins = 16;

%% Metodo matchFeatures
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));

rows = 2;
cols = 5;
numSubplots = rows * cols;

numDescriptors = 30;
main_SIFT_features = {};
main_SIFT_points = {};

function [uniqueFeatures, uniqueValidPoints] = getUniqueFeaturesAndPoints(grayImg)
    points = detectSIFTFeatures(grayImg);
    strongestPoints = selectStrongest(points, 30);
    [features, validPoints] = extractFeatures(grayImg, strongestPoints);

    % Obtener ubicaciones únicas de validPoints
    [uniqueLocations, uniqueIdx] = unique(validPoints.Location, 'rows', 'stable');
    uniqueValidPoints = validPoints(uniqueIdx);
    uniqueFeatures = features(uniqueIdx, :);

    % Seleccionar los primeros 30 puntos únicos
    if size(uniqueValidPoints, 1) > 30
        uniqueValidPoints = uniqueValidPoints(1:30, :);
        uniqueFeatures = uniqueFeatures(1:30, :);
    end
end


siftImages = {}; % Vector to store SIFT images

for i = 1:length(trainSubfolders)
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));

    for j = 1:length(imageFiles)
        if contains(imageFiles(j).name, 'sift_')
            imgPath = fullfile(folderPath, imageFiles(j).name);
            img = imread(imgPath);
            grayImg = rgb2gray(img);

            [uniqueFeatures, uniqueValidPoints] = getUniqueFeaturesAndPoints(grayImg);

            main_SIFT_features{end+1} = uniqueFeatures;
            main_SIFT_points{end+1} = uniqueValidPoints;
            siftImages{end+1} = grayImg; % Store the image

            figure;
            subplot(1, 2, 1);
            imshow(grayImg); hold on;
            plot(uniqueValidPoints);
            title(['Valid Points: ', num2str(uniqueValidPoints.Count)]);
            hold off;

            strongestPoints = selectStrongest(uniqueValidPoints, numDescriptors);
            subplot(1, 2, 2);
            imshow(grayImg); hold on;
            plot(strongestPoints);
            title(['Strongest Points: ', num2str(strongestPoints.Count)]);
            hold off;
        end
    end
end

figure;
for i = 1:min(length(main_SIFT_points), numSubplots)
    subplot(rows, cols, i);
    imshow(siftImages{i}); hold on;
    plot(main_SIFT_points{i});
    title(['Folder: ', trainSubfolders(i).name, ', Points: ', num2str(main_SIFT_points{i}.Count)]);
    hold off;
end

%% extreure probabilitats

trainData = [];
trainLabels = [];
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));

classNames = {trainSubfolders.name};

cantidad_imagenes = 0;

for i = 1:length(trainSubfolders)
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));
    
    for j = 1:length(imageFiles)
        cantidad_imagenes = cantidad_imagenes + 1;
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);
        
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        
        histR = imhist(img(:,:,1), numBins);
        histG = imhist(img(:,:,2), numBins);
        histB = imhist(img(:,:,3), numBins);
        
        histR = histR / sum(histR);
        histG = histG / sum(histG);
        histB = histB / sum(histB);
        
        featureVector = [histR; histG; histB]';
        
        % Emmagatzemem histogrames i labels
        trainData = [trainData; featureVector];
        trainLabels = [trainLabels; i];
    end
end
disp(cantidad_imagenes)


%% Carregar dades d'entrenament
trainData = [];
trainLabels = [];
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));

classNames = {trainSubfolders.name};

keypoints_bons = 0;
cap_ketpoint = 0;
for i = 1:length(trainSubfolders)
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));
    
    % verificació de que l'assignació de labels es correcta
    % fprintf('Name: %s, i: %d\n', folderPath, i);
    
    for j = 1:length(imageFiles)
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);
        grayImg = rgb2gray(img);
        
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        
        hsvImg = rgb2hsv(img);
            
        histH = imhist(hsvImg(:,:,1), numBins);
        histS = imhist(hsvImg(:,:,2), numBins);
        histV = imhist(hsvImg(:,:,3), numBins);
        
        histH = histH / sum(histH);
        histS = histS / sum(histS);
        histV = histV / sum(histV);

        [uniqueFeatures, uniqueValidPoints] = getUniqueFeaturesAndPoints(grayImg);
        
        similarityVector = zeros(1, length(main_SIFT_features));
        count = 0;
        correcte = false;
        for k = 1:length(main_SIFT_features)
            indexPairs = matchFeatures(uniqueFeatures, main_SIFT_features{k});
            if size(indexPairs, 1) == 0
                count = count + 1;
            else 
                if i == k
                    correcte = true;
                end
            end
            similarityVector(k) = size(indexPairs, 1);
        end
        if correcte
            keypoints_bons = keypoints_bons + 1;
        end
        if count < 10
            similarityVector = similarityVector / sum(similarityVector);
            cap_ketpoint = cap_ketpoint + 1;
        end    
        
        featureVector = [histH; histS; histV; similarityVector']';
        
        % Emmagatzemem histogrames i labels
        trainData = [trainData; featureVector];
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

    % fprintf('Name: %s, i: %d\n', folderPath, i);

    for j = 1:length(imageFiles)
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);
        grayImg = rgb2gray(img);
        
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        
        hsvImg = rgb2hsv(img);
            
        histH = imhist(hsvImg(:,:,1), numBins);
        histS = imhist(hsvImg(:,:,2), numBins);
        histV = imhist(hsvImg(:,:,3), numBins);
        
        histH = histH / sum(histH);
        histS = histS / sum(histS);
        histV = histV / sum(histV);

        [uniqueFeatures, uniqueValidPoints] = getUniqueFeaturesAndPoints(grayImg);
        
        similarityVector = zeros(1, length(main_SIFT_features));
        count = 0;
        for k = 1:length(main_SIFT_features)
            indexPairs = matchFeatures(uniqueFeatures, main_SIFT_features{k});
            if size(indexPairs, 1) == 0
                count = count + 1;
            end
            similarityVector(k) = size(indexPairs, 1);
        end
        if count < 10
            similarityVector = similarityVector / sum(similarityVector);
        end  
        
        featureVector = [histH; histS; histV; similarityVector']';
        
        testData = [testData; featureVector];
        testLabels = [testLabels; i];
    end
    disp(['Finished processing ', testSubfolders(i).name]);
end

%% Entrenar model Random Forest
numTrees = 100;
mdl = TreeBagger(numTrees, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');

% Evaluar model
predictedLabels = str2double(predict(mdl, testData));
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

