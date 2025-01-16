% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';

% Numero de bins
numBins = 16;

%% Carregar dades d'entrenament amb Grid Search per LBP
trainDataGrid = {};
trainLabelsGrid = {};
numNeighborsOptions = [8, 16, 24]; % Diferents opcions per numNeighbors
radiusOptions = [1, 2, 3]; % Diferents opcions per radius
fixedSize = [128 128]; % Fixed size for scaling images

trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));
classNames = {trainSubfolders.name};

for numNeighbors = numNeighborsOptions
    for radius = radiusOptions
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
                
                % Extreure característiques LBP
                lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', numNeighbors, 'Radius', radius);
                
                % Emmagatzemem les característiques LBP i labels
                trainData = [trainData; lbpFeatures];
                trainLabels = [trainLabels; i];
            end
            disp(['Finished processing ', trainSubfolders(i).name, ' with numNeighbors=', numNeighbors, ' and radius=', radius]);
        end
        % Guardar dades d'entrenament per aquesta combinació de paràmetres
        trainDataGrid{end+1} = trainData;
        trainLabelsGrid{end+1} = trainLabels;
    end
end

%% Carregar dades de test amb Grid Search per LBP
testDataGrid = {};
testLabelsGrid = {};

testSubfolders = dir(testFolder);
testSubfolders = testSubfolders([testSubfolders.isdir] & ~ismember({testSubfolders.name}, {'.', '..'}));

for numNeighbors = numNeighborsOptions
    for radius = radiusOptions
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
                
                % Extreure característiques LBP
                lbpFeatures = extractLBPFeatures(grayImg, 'NumNeighbors', numNeighbors, 'Radius', radius);
                
                % Emmagatzemem les característiques LBP i labels
                testData = [testData; lbpFeatures];
                testLabels = [testLabels; i];
            end
            disp(['Finished processing ', testSubfolders(i).name, ' with numNeighbors=', numNeighbors, ' and radius=', radius]);
        end
        % Guardar dades de test per aquesta combinació de paràmetres
        testDataGrid{end+1} = testData;
        testLabelsGrid{end+1} = testLabels;
    end
end

%% Buscar la mejor combinación de parámetros
bestAccuracy = 0;
bestParams = struct('numNeighbors', 0, 'radius', 0, 'k', 0);

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
            bestParams.numNeighbors = numNeighborsOptions(floor((idx-1)/length(radiusOptions)) + 1);
            bestParams.radius = radiusOptions(mod(idx-1, length(radiusOptions)) + 1);
            bestParams.k = k;
        end
    end
end

fprintf('Mejor combinación: numNeighbors = %d, radius = %d, k = %d con una precisión de %.2f%%\n', bestParams.numNeighbors, bestParams.radius, bestParams.k, bestAccuracy * 100);

% Entrenar y evaluar KNN con la mejor combinación de parámetros
index = (find(numNeighborsOptions == bestParams.numNeighbors) - 1) * length(radiusOptions) + find(radiusOptions == bestParams.radius);
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
bestParams = struct('numNeighbors', 0, 'radius', 0);

for idx = 1:length(trainDataGrid)
    trainData = trainDataGrid{idx};
    trainLabels = trainLabelsGrid{idx};
    testData = testDataGrid{idx};
    testLabels = testLabelsGrid{idx};
    
    % Entrenar model Random Forest
    mdl = TreeBagger(100, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');
    
    % Evaluar model
    predictedLabels = str2double(predict(mdl, testData));
    accuracy = sum(predictedLabels == testLabels) / length(testLabels);
    
    % Comprobar si es la mejor precisión
    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestParams.numNeighbors = numNeighborsOptions(floor((idx-1)/length(radiusOptions)) + 1);
        bestParams.radius = radiusOptions(mod(idx-1, length(radiusOptions)) + 1);
    end
end

fprintf('Mejor combinación: numNeighbors = %d, radius = %d con una precisión de %.2f%%\n', bestParams.numNeighbors, bestParams.radius, bestAccuracy * 100);

