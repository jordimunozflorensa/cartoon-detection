% redactar tot al overleaf
% arreglar els fitxers usats (split, post_split, posar negres, sift)
% ficar grafics xules
% mirar de guardar el model entrenat per no haver de fer-ho cada cop
% preparar un codi pq pilli el model, una image i faigui la predicció
% exploració de hiperparàmetres en bins i models de prediccio
% fer análisis dels descriptors usats (fer una mitjana dels RGB i dels 10 SIFTs)

%% pokemon
im = imread('DATA/train/pokemon/Pokemon756.jpg');

figure; imshow(im);
im(end-50:end, 1:85, :) = 0;
figure; imshow(im);

imwrite(im, 'C:/Users/jordi/Documents/main/vc/practica/DATA/train/pokemon/sift_Pokemon756.jpg');

%% southpark
im = imread('DATA/train/southpark/02499.jpg');

figure; imshow(im);

[rows, cols, ~] = size(im);
[xx, yy] = meshgrid(1:cols, 1:rows);
centerX = cols / 2;
centerY = rows / 2;
radius = min(rows, cols) / 3;
mask = (xx - centerX).^2 + (yy - centerY).^2 <= radius^2;
mask(floor(rows/2)+1:end, :) = 0; % Only apply to the upper half

im_original = im;

% Meitat superior
im(repmat(~mask, [1, 1, 3])) = 0;
% Combinació
im(floor(rows/2)+1:end, :, :) = im_original(floor(rows/2)+1:end, :, :);

figure; imshow(im);

im(end-479:end, 1:225, :) = 0;
im(end-479:end, 630:end, :) = 0;
im(end-479:80, :, :) = 0;
figure; imshow(im);

imwrite(im, 'C:/Users/jordi/Documents/main/vc/practica/DATA/train/southpark/sift_02499.jpg');
im = imread('DATA/train/southpark/sift_02499.jpg');

figure; imshow(im);