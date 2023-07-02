clc
clear all
close all
%% Training Features and Labels
imdsTrain = imageDatastore('train',...
"IncludeSubfolders",true,"FileExtensions",".jpg","LabelSource","foldernames");
Train_AlexNet_Features=[];
Train_AlexNet_Labels=[];
net=alexnet;
inputSize = net.Layers(1).InputSize;
layer='fc7';
for i=1:length(imdsTrain.Files)
    img = readimage(imdsTrain,i);
    if size(img,3)~=3
        img=cat(3,img,img,img);
    else
        img=img;
    end
    img = imresize(img, [inputSize(1) inputSize(2)]);
    F= activations(net,img,layer,'OutputAs','rows');
    L=imdsTrain.Labels(i);
    Train_AlexNet_Features=[Train_AlexNet_Features;F];
    Train_AlexNet_Labels=[Train_AlexNet_Labels;L];
end
save("Train_AlexNet_Features","Train_AlexNet_Features");
save("Train_AlexNet_Labels","Train_AlexNet_Labels");

%% Testing Features and Labels
imdsTest = imageDatastore('test',...
"IncludeSubfolders",true,"FileExtensions",".jpg","LabelSource","foldernames");
Test_AlexNet_Features=[];
Test_AlexNet_Labels=[];
for i=1:length(imdsTest.Files)
    img = readimage(imdsTest,i);
    if size(img,3)~=3
        img=cat(3,img,img,img);
    else
        img=img;
    end
    img = imresize(img, [inputSize(1) inputSize(2)]);
    F= activations(net,img,layer,'OutputAs','rows');
    L=imdsTest.Labels(i);
    Test_AlexNet_Features=[Test_AlexNet_Features;F];
    Test_AlexNet_Labels=[Test_AlexNet_Labels;L];
end
save("Test_AlexNet_Features","Test_AlexNet_Features");
save("Test_AlexNet_Labels","Test_AlexNet_Labels");
