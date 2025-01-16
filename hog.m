% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';

% Numero de bins
numBins = 16;

%% Carregar dades d'entrenament amb Grid Search
trainDataGrid = {};
trainLabelsGrid = {};
numBinsOptions = [6, 9, 12]; % Diferents opcions per numBins
cellSizeOptions = [8, 16, 32]; % Diferents opcions per cellSize
fixedSize = [128 128]; % Fixed size for scaling images

trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));
classNames = {trainSubfolders.name};

for numBins = numBinsOptions
    for cellSize = cellSizeOptions
        trainData = [];
        trainLabels = [];
        for i = 1:length(trainSubfolders)
            folderPath = fullfile(trainFolder, trainSubfolders(i).name);
            imageFiles = dir(fullfile(folderPath, '*.jpg'));
            
            for j = 1:length(imageFiles)
                imgPath = fullfile(folderPath, imageFiles(j).name);
                img = imread(imgPath);
                
                if size(img, 3) ~= 3
                    img = cat(3, img, img, img);
                end
                
                % Resize image to fixed size
                img = imresize(img, fixedSize);
                
                % Convertir a escala de grisos
                grayImg = rgb2gray(img);
                
                % Extreure característiques HOG
                [featureVector, visualization] = extractHOGFeatures(grayImg, 'CellSize', [cellSize cellSize], 'NumBins', numBins);
                
                % Emmagatzemem les característiques HOG i labels
                trainData = [trainData; featureVector];
                trainLabels = [trainLabels; i];
            end
            disp(['Finished processing ', trainSubfolders(i).name, ' with numBins=', numBins, ' and cellSize=', cellSize]);
        end
        % Guardar dades d'entrenament per aquesta combinació de paràmetres
        trainDataGrid{end+1} = trainData;
        trainLabelsGrid{end+1} = trainLabels;
    end
end

%% Carregar dades de test
testDataGrid = {};
testLabelsGrid = {};

testSubfolders = dir(testFolder);
testSubfolders = testSubfolders([testSubfolders.isdir] & ~ismember({testSubfolders.name}, {'.', '..'}));

for numBins = numBinsOptions
    for cellSize = cellSizeOptions
        testData = [];
        testLabels = [];
        for i = 1:length(testSubfolders)
            folderPath = fullfile(testFolder, testSubfolders(i).name);
            imageFiles = dir(fullfile(folderPath, '*.jpg'));

            for j = 1:length(imageFiles)
                imgPath = fullfile(folderPath, imageFiles(j).name);
                img = imread(imgPath);
                
                if size(img, 3) ~= 3
                    img = cat(3, img, img, img);
                end
                
                % Resize image to fixed size
                img = imresize(img, fixedSize);
                
                % Convertir a escala de grisos
                grayImg = rgb2gray(img);
                
                % Extreure característiques HOG
                [featureVector, visualization] = extractHOGFeatures(grayImg, 'CellSize', [cellSize cellSize], 'NumBins', numBins);
                
                % Emmagatzemem les característiques HOG i labels
                testData = [testData; featureVector];
                testLabels = [testLabels; i];
            end
            disp(['Finished processing ', testSubfolders(i).name, ' with numBins=', numBins, ' and cellSize=', cellSize]);
        end
        % Guardar dades de test per aquesta combinació de paràmetres
        testDataGrid{end+1} = testData;
        testLabelsGrid{end+1} = testLabels;
    end
end

%% Buscar la mejor combinación de parámetros
bestAccuracy = 0;
bestParams = struct('numBins', 0, 'cellSize', 0, 'k', 0);

kValues = 1:10;

for idx = 1:length(trainDataGrid)
    trainData = trainDataGrid{idx};
    trainLabels = trainLabelsGrid{idx};
    testData = testDataGrid{idx};
    testLabels = testLabelsGrid{idx};
    
    for k = kValues
        % Entrenar model KNN
        MdlKNN = fitcknn(trainData, trainLabels, 'NumNeighbors', k);
        
        % Evaluar model
        predictedLabels = predict(MdlKNN, testData);
        accuracy = sum(predictedLabels == testLabels) / length(testLabels);
        
        % Comprobar si es la mejor precisión
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestParams.numBins = numBinsOptions(floor((idx-1)/length(cellSizeOptions)) + 1);
            bestParams.cellSize = cellSizeOptions(mod(idx-1, length(cellSizeOptions)) + 1);
            bestParams.k = k;
        end
    end
end

fprintf('Mejor combinación: numBins = %d, cellSize = %d, k = %d con una precisión de %.2f%%\n', bestParams.numBins, bestParams.cellSize, bestParams.k, bestAccuracy * 100);

% Entrenar y evaluar KNN con la mejor combinación de parámetros
index = (find(numBinsOptions == bestParams.numBins) - 1) * length(cellSizeOptions) + find(cellSizeOptions == bestParams.cellSize);
trainData = trainDataGrid{index};
trainLabels = trainLabelsGrid{index};
testData = testDataGrid{index};
testLabels = testLabelsGrid{index};

MdlKNN = fitcknn(trainData, trainLabels, 'NumNeighbors', bestParams.k);
predictedLabels = predict(MdlKNN, testData);
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del KNN con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per KNN
figure;
confMatKNN = confusionmat(testLabels, predictedLabels);
confusionchart(confMatKNN, classNames);
title('Matriu de confusió per KNN');

%% Buscar la mejor combinación de parámetros
bestAccuracy = 0;
bestParams = struct('numBins', 0, 'cellSize', 0);

for idx = 1:length(trainDataGrid)
    trainData = trainDataGrid{idx};
    trainLabels = trainLabelsGrid{idx};
    
    % Entrenar model Random Forest
    mdl = TreeBagger(100, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');
    
    % Evaluar model
    predictedLabels = str2double(predict(mdl, testDataGrid{idx}));
    accuracy = sum(predictedLabels == testLabelsGrid{idx}) / length(testLabelsGrid{idx});
    
    % Comprobar si es la mejor precisión
    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestParams.numBins = numBinsOptions(floor((idx-1)/length(cellSizeOptions)) + 1);
        bestParams.cellSize = cellSizeOptions(mod(idx-1, length(cellSizeOptions)) + 1);
    end
end

fprintf('Mejor combinación: numBins = %d, cellSize = %d con una precisión de %.2f%%\n', bestParams.numBins, bestParams.cellSize, bestAccuracy * 100);