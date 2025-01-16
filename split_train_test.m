% Configuración
baseFolders = {'TRAIN'}; % Carpetas principales con las series
outputFolder = 'DATA'; % Carpeta de salida para train y test
trainRatio = 0.7; % Porcentaje de imágenes para entrenamiento

% Crear carpetas de salida
trainFolder = fullfile(outputFolder, 'train');
testFolder = fullfile(outputFolder, 'test');
if ~exist(trainFolder, 'dir')
    mkdir(trainFolder);
end
if ~exist(testFolder, 'dir')
    mkdir(testFolder);
end

% Procesar cada carpeta base
for k = 1:length(baseFolders)
    baseFolder = baseFolders{k};
    
    % Obtener subcarpetas
    subfolders = dir(baseFolder);
    subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

    for i = 1:length(subfolders)
        seriesName = subfolders(i).name;
        seriesFolder = fullfile(baseFolder, seriesName);
        imageFiles = dir(fullfile(seriesFolder, '*.jpg')); % Cambia la extensión según tus imágenes
        numImages = length(imageFiles);

        % Crear carpetas específicas para la serie en train y test
        trainSeriesFolder = fullfile(trainFolder, seriesName);
        testSeriesFolder = fullfile(testFolder, seriesName);
        if ~exist(trainSeriesFolder, 'dir')
            mkdir(trainSeriesFolder);
        end
        if ~exist(testSeriesFolder, 'dir')
            mkdir(testSeriesFolder);
        end

        % Barajar y dividir las imágenes
        indices = randperm(numImages);
        numTrain = round(trainRatio * numImages);

        trainIndices = indices(1:numTrain);
        testIndices = indices(numTrain+1:end);

        % Copiar imágenes a las carpetas correspondientes
        for j = trainIndices
            sourceFile = fullfile(seriesFolder, imageFiles(j).name);
            destinationFile = fullfile(trainSeriesFolder, imageFiles(j).name);
            copyfile(sourceFile, destinationFile);
        end

        for j = testIndices
            sourceFile = fullfile(seriesFolder, imageFiles(j).name);
            destinationFile = fullfile(testSeriesFolder, imageFiles(j).name);
            copyfile(sourceFile, destinationFile);
        end
    end
end

fprintf('División completada. Las carpetas de entrenamiento y prueba se encuentran en: %s\n', outputFolder);