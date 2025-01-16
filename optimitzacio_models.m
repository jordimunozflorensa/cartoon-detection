% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';

fixedSize = [128 128];

% parametres KNN
numBinsColorKnn = 16;
numBinsHOGKnn = 9;
cellSizeKnn = 32;
veinsLBPKnn = 8;
radiLBPKnn = 1;
veins = 1;

%% Carregar dades d'entrenament
trainDataColor = [];
trainDataLBP = [];
trainDataHOG = [];
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

        grayImg = rgb2gray(img);
        
        histR = imhist(img(:,:,1), numBinsColorKnn);
        histG = imhist(img(:,:,2), numBinsColorKnn);
        histB = imhist(img(:,:,3), numBinsColorKnn);
        
        histR = histR / sum(histR);
        histG = histG / sum(histG);
        histB = histB / sum(histB);
        
        colorHistoFeatures = [histR; histG; histB]';
        lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', veinsLBPKnn, 'Radius', radiLBPKnn);
        [hogFeatures, visualization] = extractHOGFeatures(grayImg, 'CellSize', [cellSizeKnn cellSizeKnn], 'NumBins', numBinsHOGKnn);

        trainDataColor = [trainDataColor; colorHistoFeatures];
        trainDataLBP = [trainDataLBP; lbpFeatures];
        trainDataHOG = [trainDataHOG; hogFeatures];
        trainLabels = [trainLabels; i];
    end
    disp(['Finished processing ', trainSubfolders(i).name]);
end

%% Carregar dades de test
testDataColor = [];
testDataLBP = [];
testDataHOG = [];
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
        
        histR = imhist(img(:,:,1), numBinsColorKnn);
        histG = imhist(img(:,:,2), numBinsColorKnn);
        histB = imhist(img(:,:,3), numBinsColorKnn);
        
        histR = histR / sum(histR);
        histG = histG / sum(histG);
        histB = histB / sum(histB);
        
        colorHistoFeatures = [histR; histG; histB]';
        lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', veinsLBPKnn, 'Radius', radiLBPKnn);
        [hogFeatures, visualization] = extractHOGFeatures(grayImg, 'CellSize', [cellSizeKnn cellSizeKnn], 'NumBins', numBinsHOGKnn);

        testDataColor = [testDataColor; colorHistoFeatures];
        testDataLBP = [testDataLBP; lbpFeatures];
        testDataHOG = [testDataHOG; hogFeatures];
        testLabels = [testLabels; i];
    end
    disp(['Finished processing ', testSubfolders(i).name]);
    
end

%% Entrenar models

conjuntosTrain = {trainDataColor, trainDataLBP, trainDataHOG};
conjuntosTest = {testDataColor, testDataLBP, testDataHOG};
labelsTrain = categorical(trainLabels); % Convertir etiquetas a categórico si no lo están
labelsTest = categorical(testLabels);   % Convertir etiquetas a categórico si no lo están

mejoresPrecisiones = zeros(2, 3); % Almacena las mejores precisiones para cada modelo y conjunto

modelos = {'KNN', 'TreeBagger'};
nombresConjuntos = {'Color', 'LBP', 'HOG'};

for i = 1:length(conjuntosTrain)
    trainData = conjuntosTrain{i};
    testData = conjuntosTest{i};

    % KNN
    disp(['Optimizando KNN para el conjunto ', nombresConjuntos{i}]);
    knnParams = hyperparameters('fitcknn', trainData, labelsTrain);
    knnParams(1).Range = [1, 20]; % Número de vecinos

    optKNN = fitcknn(trainData, labelsTrain, 'OptimizeHyperparameters', 'all', ...
                     'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 0));

    predKNN = predict(optKNN, testData);
    accKNN = mean(predKNN == labelsTest);
    mejoresPrecisiones(1, i) = accKNN;

    % TreeBagger
    disp(['Optimizando TreeBagger para el conjunto ', nombresConjuntos{i}]);
    numTrees = [10, 50, 100, 200];
    maxFeaturesOptions = {'sqrt', 'log2', 'all'}; % Opciones

    bestAccTree = 0;

    for n = numTrees
        for f = 1:length(maxFeaturesOptions)
            % Calcular el número de predictores basado en maxFeaturesOptions
            if strcmp(maxFeaturesOptions{f}, 'sqrt')
                numPredictors = ceil(sqrt(size(trainData, 2)));
            elseif strcmp(maxFeaturesOptions{f}, 'log2')
                numPredictors = ceil(log2(size(trainData, 2)));
            elseif strcmp(maxFeaturesOptions{f}, 'all')
                numPredictors = 'all';
            end

            % Entrenar TreeBagger con la opción calculada
            tree = TreeBagger(n, trainData, labelsTrain, 'Method', 'classification', ...
                            'NumPredictorsToSample', numPredictors);

            % Predecir en el conjunto de prueba
            predTree = str2double(predict(tree, testData));
            accTree = mean(categorical(predTree) == labelsTest);

            % Actualizar la mejor precisión
            if accTree > bestAccTree
                bestAccTree = accTree;
            end
        end
    end

    mejoresPrecisiones(2, i) = bestAccTree;
end

% Resultados
disp('Mejores precisiones obtenidas para cada modelo y conjunto:');
for i = 1:3
    disp(['Conjunto ', nombresConjuntos{i}, ':']);
    for j = 1:2
        disp([modelos{j}, ': ', num2str(mejoresPrecisiones(j, i))]);
    end
end

%%
% Mostrar los mejores hiperparámetros obtenidos
disp('Mejores hiperparámetros obtenidos para cada modelo y conjunto:');

% KNN
disp('KNN:');
disp(['Color: ', 'NumNeighbors = ', num2str(optKNN.NumNeighbors)]);
disp(['LBP: ', 'NumNeighbors = ', num2str(optKNN.NumNeighbors)]);
disp(['HOG: ', 'NumNeighbors = ', num2str(optKNN.NumNeighbors)]);

% TreeBagger
disp('TreeBagger:');
disp(['Color: ', 'NumTrees = ', num2str(tree.NumTrees), ', NumPredictorsToSample = ', num2str(tree.NumPredictorsToSample)]);
disp(['LBP: ', 'NumTrees = ', num2str(tree.NumTrees), ', NumPredictorsToSample = ', num2str(tree.NumPredictorsToSample)]);
disp(['HOG: ', 'NumTrees = ', num2str(tree.NumTrees), ', NumPredictorsToSample = ', num2str(tree.NumPredictorsToSample)]);