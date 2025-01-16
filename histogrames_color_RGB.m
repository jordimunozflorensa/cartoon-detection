% Carpetes
trainFolder = 'DATA/train';
testFolder = 'DATA/test';

% Numero de bins
numBins = 16;

%% Visualització dels histogrames de color
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));
classNames = {trainSubfolders.name};

figure;
rows = 5;
cols = 2;
numSubplots = rows * cols;

% recorrem totes les carpetes
for i = 1:length(trainSubfolders)
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));

    % Acumular histogrames de las primeres N imatges per no utilitzar nomes
    % un
    numImagesToVisualize = min(5, length(imageFiles));
    aggregateHistR = zeros(numBins, 1);
    aggregateHistG = zeros(numBins, 1);
    aggregateHistB = zeros(numBins, 1);
    
    % recorrem totes les imatges de la carpeta
    for j = 1:numImagesToVisualize
        imgPath = fullfile(folderPath, imageFiles(j).name);
        img = imread(imgPath);

        % Convertir a espai de color RGB, per si despres afegim imatges
        % rares
        if size(img, 3) ~= 3
            img = cat(3, img, img, img);
        end

        % Histograma de color
        histR = imhist(img(:,:,1), numBins);
        histG = imhist(img(:,:,2), numBins);
        histB = imhist(img(:,:,3), numBins);

        % Normalitzar histograma
        histR = histR / sum(histR);

        histB = histB / sum(histB);

        % Acumular histograma
        aggregateHistR = aggregateHistR + histR;
        aggregateHistG = aggregateHistG + histG;
        aggregateHistB = aggregateHistB + histB;
    end

    % AVG histogrames
    aggregateHistR = aggregateHistR / numImagesToVisualize;
    aggregateHistG = aggregateHistG / numImagesToVisualize;
    aggregateHistB = aggregateHistB / numImagesToVisualize;

    % Mostrar histogrames
    subplot(rows, cols, i);
    bar(1:numBins, [aggregateHistR, aggregateHistG, aggregateHistB], 'stacked');
    xlabel('Bins');
    ylabel('Frecuencia Norm');
    title(['Serie: ', trainSubfolders(i).name]);
    legend({'Rojo', 'Verde', 'Azul'});
end

%% Carregar dades d'entrenament
trainData = [];
trainLabels = [];
trainSubfolders = dir(trainFolder);
trainSubfolders = trainSubfolders([trainSubfolders.isdir] & ~ismember({trainSubfolders.name}, {'.', '..'}));

classNames = {trainSubfolders.name};

for i = 1:length(trainSubfolders)
    folderPath = fullfile(trainFolder, trainSubfolders(i).name);
    imageFiles = dir(fullfile(folderPath, '*.jpg'));
    
    % verificació de que l'assignació de labels es correcta
    % fprintf('Name: %s, i: %d\n', folderPath, i);
    
    for j = 1:length(imageFiles)
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
        
        testData = [testData; featureVector];
        testLabels = [testLabels; i];
    end
end

%% Entrenar model Random Forest
numTrees = 100;
mdl = TreeBagger(numTrees, trainData, trainLabels, 'OOBPrediction', 'On', 'Method', 'classification');

% Evaluar model
predictedLabels = str2double(predict(mdl, testData));
accuracy = sum(predictedLabels == testLabels) / length(testLabels);

fprintf('Precisión del modelo: %.2f%%\n', accuracy * 100);

%% Mostrar evolució del nombre d'arbres
figure;
oobError = oobError(mdl);
plot(oobError);
xlabel('Número darbres');
ylabel('Error');
title('Evolució del error OOB');

%% Matriu de confusió
figure;
confMat = confusionmat(testLabels, predictedLabels);
confusionchart(confMat, classNames);
title('Matriu de confusió');

