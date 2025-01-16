% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';
fixedSize = [128 128];

%% Visualització dels histogrames de color
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));
classNames = {trainSubfolders.name};
numBins = 16;
figure;
rows = 5;
cols = 3;
numSubplots = rows * cols;

% recorrem totes les carpetes
for i = 6:10
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));

    % Acumular histogrames de las primeres N imatges per no utilitzar nomes
    % un
    numImagesToVisualize = min(5, length(imageFiles));
    aggregateHistH = zeros(numBins, 1);
    aggregateHistS = zeros(numBins, 1);
    aggregateHistV = zeros(numBins, 1);
    
    % recorrem totes les imatges de la carpeta
    for j = 1:numImagesToVisualize
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);

        % Convertir a espai de color HSV
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        hsvImg = rgb2hsv(img);

        % Histograma de color
        histH = imhist(hsvImg(:,:,1), numBins);
        histS = imhist(hsvImg(:,:,2), numBins);
        histV = imhist(hsvImg(:,:,3), numBins);

        % Normalitzar histograma
        histH = histH / sum(histH);
        histS = histS / sum(histS);
        histV = histV / sum(histV);

        % Acumular histograma
        aggregateHistH = aggregateHistH + histH;
        aggregateHistS = aggregateHistS + histS;
        aggregateHistV = aggregateHistV + histV;
    end

    % AVG histogrames
    aggregateHistH = aggregateHistH / numImagesToVisualize;
    aggregateHistS = aggregateHistS / numImagesToVisualize;
    aggregateHistV = aggregateHistV / numImagesToVisualize;

    % Mostrar histogrames
    subplot(rows, cols, (i-6)*3 + 1);
    bar(1:numBins, aggregateHistH);
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Hue: ', trainSubfolders(i).name]);

    subplot(rows, cols, (i-6)*3 + 2);
    bar(1:numBins, aggregateHistS);
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Saturation: ', trainSubfolders(i).name]);

    subplot(rows, cols, (i-6)*3 + 3);
    bar(1:numBins, aggregateHistV);
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Value: ', trainSubfolders(i).name]);
end

figure;
rows = 5;
cols = 3;
numSubplots = rows * cols;
for i = 1:5
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));

    % Acumular histogrames de las primeres N imatges per no utilitzar nomes
    % un
    numImagesToVisualize = min(5, length(imageFiles));
    aggregateHistH = zeros(numBins, 1);
    aggregateHistS = zeros(numBins, 1);
    aggregateHistV = zeros(numBins, 1);

    % recorrem totes les imatges de la carpeta
    for j = 1:numImagesToVisualize
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);

        % Convertir a espai de color HSV
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end
        hsvImg = rgb2hsv(img);

        % Histograma de color
        histH = imhist(hsvImg(:,:,1), numBins);
        histS = imhist(hsvImg(:,:,2), numBins);
        histV = imhist(hsvImg(:,:,3), numBins);

        % Normalitzar histograma
        histH = histH / sum(histH);
        histS = histS / sum(histS);
        histV = histV / sum(histV);

        % Acumular histograma
        aggregateHistH = aggregateHistH + histH;
        aggregateHistS = aggregateHistS + histS;
        aggregateHistV = aggregateHistV + histV;
    end

    % AVG histogrames
    aggregateHistH = aggregateHistH / numImagesToVisualize;
    aggregateHistS = aggregateHistS / numImagesToVisualize;
    aggregateHistV = aggregateHistV / numImagesToVisualize;

    % Mostrar histogrames
    subplot(rows, cols, (i-1)*3 + 1);
    bar(1:numBins, aggregateHistH);
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Hue: ', trainSubfolders(i).name]);

    subplot(rows, cols, (i-1)*3 + 2);
    bar(1:numBins, aggregateHistS);
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Saturation: ', trainSubfolders(i).name]);

    subplot(rows, cols, (i-1)*3 + 3);
    bar(1:numBins, aggregateHistV);
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Value: ', trainSubfolders(i).name]);
end

%% Carregar dades d'entrenament
trainDataGridHSV = {};
trainLabelsGridHSV = {};
trainDataGridRGB = {};
trainLabelsGridRGB = {};
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));

classNames = {trainSubfolders.name};

numBinsOptions = [16]; % Diferents valors de numBins per explorar

for numBins = numBinsOptions
    trainDataRGB = [];
    trainLabelsRGB = [];
    trainDataHSV = [];
    trainLabelsHSV = [];
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
            
            histH = imhist(hsvImg(:,:,1), numBins);
            histS = imhist(hsvImg(:,:,2), numBins);
            histV = imhist(hsvImg(:,:,3), numBins);
            
            histH = histH / sum(histH);
            histS = histS / sum(histS);
            histV = histV / sum(histV);
            
            HSVcolorHistoFeatures = [histH; histS; histV]';
                
            histR = imhist(img(:,:,1), numBins);
            histG = imhist(img(:,:,2), numBins);
            histB = imhist(img(:,:,3), numBins);
            
            histR = histR / sum(histR);
            histG = histG / sum(histG);
            histB = histB / sum(histB);
            
            RGBcolorHistoFeatures = [histR; histG; histB]';
            
            % Emmagatzemem histogrames i labels
            trainDataHSV = [trainDataHSV; HSVcolorHistoFeatures];
            trainDataRGB = [trainDataRGB; RGBcolorHistoFeatures];
            trainLabelsHSV = [trainLabelsHSV; i];
            trainLabelsRGB = [trainLabelsRGB; i];
            
        end
    end
    % Guardar dades d'entrenament per aquesta combinació de paràmetres
    trainDataGridHSV{end+1} = trainDataHSV;
    trainLabelsGridHSV{end+1} = trainLabelsHSV;
    trainDataGridRGB{end+1} = trainDataRGB;
    trainLabelsGridRGB{end+1} = trainLabelsRGB;
end

%% Carregar dades de test
testDataGridHSV = {};
testLabelsGridHSV = {};
testDataGridRGB = {};
testLabelsGridRGB = {};
testSubfolders = dir(testFolder);
testSubfolders = testSubfolders([testSubfolders.isdir] & ~ismember({testSubfolders.name}, {'.', '..'}));

for numBins = numBinsOptions
    testDataRGB = [];
    testLabelsRGB = [];
    testDataHSV = [];
    testLabelsHSV = [];
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

            hsvImg = rgb2hsv(img);
            
            histH = imhist(hsvImg(:,:,1), numBins);
            histS = imhist(hsvImg(:,:,2), numBins);
            histV = imhist(hsvImg(:,:,3), numBins);
            
            histH = histH / sum(histH);
            histS = histS / sum(histS);
            histV = histV / sum(histV);
            
            HSVcolorHistoFeatures = [histH; histS; histV]';
                
            histR = imhist(img(:,:,1), numBins);
            histG = imhist(img(:,:,2), numBins);
            histB = imhist(img(:,:,3), numBins);
            
            histR = histR / sum(histR);
            histG = histG / sum(histG);
            histB = histB / sum(histB);
            
            RGBcolorHistoFeatures = [histR; histG; histB]';
            
            % Emmagatzemem histogrames i labels
            testDataHSV = [testDataHSV; HSVcolorHistoFeatures];
            testDataRGB = [testDataRGB; RGBcolorHistoFeatures];
            testLabelsHSV = [testLabelsHSV; i];
            testLabelsRGB = [testLabelsRGB; i];
        end
    end
    % Guardar dades de test per aquesta combinació de paràmetres
    testDataGridHSV{end+1} = testDataHSV;
    testLabelsGridHSV{end+1} = testLabelsHSV;
    testDataGridRGB{end+1} = testDataRGB;
    testLabelsGridRGB{end+1} = testLabelsRGB;
end

%% Buscar la mejor combinación de parámetros
% Buscar la mejor combinación de parámetros para KNN en RGB
bestAccuracyKNN_RGB = 0;
bestParamsKNN_RGB = struct('numBins', 0, 'k', 0);

kValues = 1:10;

for idx = 1:length(trainDataGridRGB)
    trainData = trainDataGridRGB{idx};
    trainLabels = trainLabelsGridRGB{idx};
    testData = testDataGridRGB{idx};
    testLabels = testLabelsGridRGB{idx};
    
    for k = kValues
        % Entrenar model KNN
        MdlKNN = fitcknn(trainData, trainLabels, 'NumNeighbors', k);
        
        % Evaluar model
        predictedLabels = predict(MdlKNN, testData);
        accuracy = sum(predictedLabels == testLabels) / length(testLabels);
        
        % Comprobar si es la mejor precisión
        if accuracy > bestAccuracyKNN_RGB
            bestAccuracyKNN_RGB = accuracy;
            bestParamsKNN_RGB.numBins = numBinsOptions(idx);
            bestParamsKNN_RGB.k = k;
        end
    end
end

fprintf('Mejor combinación para KNN en RGB: numBins = %d, k = %d con una precisión de %.2f%%\n', bestParamsKNN_RGB.numBins, bestParamsKNN_RGB.k, bestAccuracyKNN_RGB * 100);

% Entrenar y evaluar KNN con la mejor combinación de parámetros en RGB
index = find(numBinsOptions == bestParamsKNN_RGB.numBins);
trainData = trainDataGridRGB{index};
trainLabels = trainLabelsGridRGB{index};
testData = testDataGridRGB{index};
testLabels = testLabelsGridRGB{index};

MdlKNN = fitcknn(trainData, trainLabels, 'NumNeighbors', bestParamsKNN_RGB.k);
predictedLabels = predict(MdlKNN, testData);
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del KNN en RGB con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per KNN en RGB
confMatKNN_RGB = confusionmat(testLabels, predictedLabels);

% Buscar la mejor combinación de parámetros para KNN en HSV
bestAccuracyKNN_HSV = 0;
bestParamsKNN_HSV = struct('numBins', 0, 'k', 0);

for idx = 1:length(trainDataGridHSV)
    trainData = trainDataGridHSV{idx};
    trainLabels = trainLabelsGridHSV{idx};
    testData = testDataGridHSV{idx};
    testLabels = testLabelsGridHSV{idx};
    
    for k = kValues
        % Entrenar model KNN
        MdlKNN = fitcknn(trainData, trainLabels, 'NumNeighbors', k);
        
        % Evaluar model
        predictedLabels = predict(MdlKNN, testData);
        accuracy = sum(predictedLabels == testLabels) / length(testLabels);
        
        % Comprobar si es la mejor precisión
        if accuracy > bestAccuracyKNN_HSV
            bestAccuracyKNN_HSV = accuracy;
            bestParamsKNN_HSV.numBins = numBinsOptions(idx);
            bestParamsKNN_HSV.k = k;
        end
    end
end

fprintf('Mejor combinación para KNN en HSV: numBins = %d, k = %d con una precisión de %.2f%%\n', bestParamsKNN_HSV.numBins, bestParamsKNN_HSV.k, bestAccuracyKNN_HSV * 100);

% Entrenar y evaluar KNN con la mejor combinación de parámetros en HSV
index = find(numBinsOptions == bestParamsKNN_HSV.numBins);
trainData = trainDataGridHSV{index};
trainLabels = trainLabelsGridHSV{index};
testData = testDataGridHSV{index};
testLabels = testLabelsGridHSV{index};

MdlKNN = fitcknn(trainData, trainLabels, 'NumNeighbors', bestParamsKNN_HSV.k);
predictedLabels = predict(MdlKNN, testData);
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del KNN en HSV con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per KNN en HSV
confMatKNN_HSV = confusionmat(testLabels, predictedLabels);

% Mostrar las matrices de confusión en un subplot
figure;
subplot(1, 2, 1);
confusionchart(confMatKNN_RGB, classNames);
title('Matriu de confusió per KNN en RGB');

subplot(1, 2, 2);
confusionchart(confMatKNN_HSV, classNames);
title('Matriu de confusió per KNN en HSV');

%% Buscar la mejor combinación de parámetros para Random Forest en RGB
bestAccuracyRF_RGB = 0;
bestParamsRF_RGB = struct('numBins', 0);

for idx = 1:length(trainDataGridRGB)
    trainData = trainDataGridRGB{idx};
    trainLabels = trainLabelsGridRGB{idx};
    testData = testDataGridRGB{idx};
    testLabels = testLabelsGridRGB{idx};clc
    
    % Entrenar model Random Forest
    MdlRF = TreeBagger(100, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');
    
    % Evaluar model
    predictedLabels = predict(MdlRF, testData);
    predictedLabels = str2double(predictedLabels); % Convertir a numérico
    accuracy = sum(predictedLabels == testLabels) / length(testLabels);
    
    % Comprobar si es la mejor precisión
    if accuracy > bestAccuracyRF_RGB
        bestAccuracyRF_RGB = accuracy;
        bestParamsRF_RGB.numBins = numBinsOptions(idx);
    end
end

fprintf('Mejor combinación para Random Forest en RGB: numBins = %d con una precisión de %.2f%%\n', bestParamsRF_RGB.numBins, bestAccuracyRF_RGB * 100);


% Entrenar y evaluar Random Forest con la mejor combinación de parámetros en RGB
index = find(numBinsOptions == bestParamsRF_RGB.numBins);
trainData = trainDataGridRGB{index};
trainLabels = trainLabelsGridRGB{index};
testData = testDataGridRGB{index};
testLabels = testLabelsGridRGB{index};

MdlRF = TreeBagger(100, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');
predictedLabels = predict(MdlRF, testData);
predictedLabels = str2double(predictedLabels); % Convertir a numérico
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del Random Forest en RGB con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per Random Forest en RGB
confMatRF_RGB = confusionmat(testLabels, predictedLabels);

% Buscar la mejor combinación de parámetros para Random Forest en HSV
bestAccuracyRF_HSV = 0;
bestParamsRF_HSV = struct('numBins', 0);

for idx = 1:length(trainDataGridHSV)
    trainData = trainDataGridHSV{idx};
    trainLabels = trainLabelsGridHSV{idx};
    testData = testDataGridHSV{idx};
    testLabels = testLabelsGridHSV{idx};
    
    % Entrenar model Random Forest
    MdlRF = TreeBagger(50, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');
    
    % Evaluar model
    predictedLabels = predict(MdlRF, testData);
    predictedLabels = str2double(predictedLabels); % Convertir a numérico
    accuracy = sum(predictedLabels == testLabels) / length(testLabels);
    
    % Comprobar si es la mejor precisión
    if accuracy > bestAccuracyRF_HSV
        bestAccuracyRF_HSV = accuracy;
        bestParamsRF_HSV.numBins = numBinsOptions(idx);
    end
end

fprintf('Mejor combinación para Random Forest en HSV: numBins = %d con una precisión de %.2f%%\n', bestParamsRF_HSV.numBins, bestAccuracyRF_HSV * 100);

% Entrenar y evaluar Random Forest con la mejor combinación de parámetros en HSV
index = find(numBinsOptions == bestParamsRF_HSV.numBins);
trainData = trainDataGridHSV{index};
trainLabels = trainLabelsGridHSV{index};
testData = testDataGridHSV{index};
testLabels = testLabelsGridHSV{index};

MdlRF = TreeBagger(50, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');
predictedLabels = predict(MdlRF, testData);
predictedLabels = str2double(predictedLabels); % Convertir a numérico
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del Random Forest en HSV con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per Random Forest en HSV
confMatRF_HSV = confusionmat(testLabels, predictedLabels);

% Mostrar las matrices de confusión en un subplot
figure;
subplot(1, 2, 1);
confusionchart(confMatRF_RGB, classNames);
title('Matriu de confusió per Random Forest en RGB');

subplot(1, 2, 2);
confusionchart(confMatRF_HSV, classNames);
title('Matriu de confusió per Random Forest en HSV');

%% Buscar la mejor combinación de parámetros para SVM en RGB
bestAccuracySVM_RGB = 0;
bestParamsSVM_RGB = struct('numBins', 0, 'KernelFunction', '');

kernelFunctions = {'linear', 'gaussian', 'polynomial'};

for idx = 1:length(trainDataGridRGB)
    trainData = trainDataGridRGB{idx};
    trainLabels = trainLabelsGridRGB{idx};
    testData = testDataGridRGB{idx};
    testLabels = testLabelsGridRGB{idx};
    
    for kernel = kernelFunctions
        % Entrenar model SVM
        MdlSVM = fitcecoc(trainData, trainLabels, 'Learners', templateSVM('KernelFunction', kernel{1}));
        
        % Evaluar model
        predictedLabels = predict(MdlSVM, testData);
        accuracy = sum(predictedLabels == testLabels) / length(testLabels);
        
        % Comprobar si es la mejor precisión
        if accuracy > bestAccuracySVM_RGB
            bestAccuracySVM_RGB = accuracy;
            bestParamsSVM_RGB.numBins = numBinsOptions(idx);
            bestParamsSVM_RGB.KernelFunction = kernel{1};
        end
    end
end

fprintf('Mejor combinación para SVM en RGB: numBins = %d, KernelFunction = %s con una precisión de %.2f%%\n', bestParamsSVM_RGB.numBins, bestParamsSVM_RGB.KernelFunction, bestAccuracySVM_RGB * 100);

% Entrenar y evaluar SVM con la mejor combinación de parámetros en RGB
index = find(numBinsOptions == bestParamsSVM_RGB.numBins);
trainData = trainDataGridRGB{index};
trainLabels = trainLabelsGridRGB{index};
testData = testDataGridRGB{index};
testLabels = testLabelsGridRGB{index};

MdlSVM = fitcecoc(trainData, trainLabels, 'Learners', templateSVM('KernelFunction', bestParamsSVM_RGB.KernelFunction));
predictedLabels = predict(MdlSVM, testData);
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del SVM en RGB con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per SVM en RGB
confMatSVM_RGB = confusionmat(testLabels, predictedLabels);

% Buscar la mejor combinación de parámetros para SVM en HSV
bestAccuracySVM_HSV = 0;
bestParamsSVM_HSV = struct('numBins', 0, 'KernelFunction', '');

for idx = 1:length(trainDataGridHSV)
    trainData = trainDataGridHSV{idx};
    trainLabels = trainLabelsGridHSV{idx};
    testData = testDataGridHSV{idx};
    testLabels = testLabelsGridHSV{idx};
    
    for kernel = kernelFunctions
        % Entrenar model SVM
        MdlSVM = fitcecoc(trainData, trainLabels, 'Learners', templateSVM('KernelFunction', kernel{1}));
        
        % Evaluar model
        predictedLabels = predict(MdlSVM, testData);
        accuracy = sum(predictedLabels == testLabels) / length(testLabels);
        
        % Comprobar si es la mejor precisión
        if accuracy > bestAccuracySVM_HSV
            bestAccuracySVM_HSV = accuracy;
            bestParamsSVM_HSV.numBins = numBinsOptions(idx);
            bestParamsSVM_HSV.KernelFunction = kernel{1};
        end
    end
end

fprintf('Mejor combinación para SVM en HSV: numBins = %d, KernelFunction = %s con una precisión de %.2f%%\n', bestParamsSVM_HSV.numBins, bestParamsSVM_HSV.KernelFunction, bestAccuracySVM_HSV * 100);

% Entrenar y evaluar SVM con la mejor combinación de parámetros en HSV
index = find(numBinsOptions == bestParamsSVM_HSV.numBins);
trainData = trainDataGridHSV{index};
trainLabels = trainLabelsGridHSV{index};
testData = testDataGridHSV{index};
testLabels = testLabelsGridHSV{index};

MdlSVM = fitcecoc(trainData, trainLabels, 'Learners', templateSVM('KernelFunction', bestParamsSVM_HSV.KernelFunction));
predictedLabels = predict(MdlSVM, testData);
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del SVM en HSV con la mejor combinación de parámetros: %.2f%%\n', accuracy * 100);

% Matriu de confusió per SVM en HSV
confMatSVM_HSV = confusionmat(testLabels, predictedLabels);

% Mostrar las matrices de confusión en un subplot
figure;
subplot(1, 2, 1);
confusionchart(confMatSVM_RGB, classNames);
title('Matriu de confusió per SVM en RGB');

subplot(1, 2, 2);
confusionchart(confMatSVM_HSV, classNames);
title('Matriu de confusió per SVM en HSV');
%%
RFModel = TreeBagger(100, trainDataHSV, trainLabelsHSV, 'OOBPredictorImportance', 'on');
importance = RFModel.OOBPermutedPredictorDeltaError;

figure;
bar(importance);
title('Feature Importance Estimates');
xlabel('Feature Index');
ylabel('Importance');