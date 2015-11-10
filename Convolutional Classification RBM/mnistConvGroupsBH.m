% The following code implements Convolutional Classification RBM as a unit. 
%
% The dataset used is handwritten digit data - MNIST
% The code calculates energy of the configuration after each epoch to check
% for convergence of algorithm.
% Dropout and sparsity is also implemented which gives better classification performance.


clear all;
load('Dataset/mnistConvData.mat')

mnistImgDim = 28;
wDim = 8;
numGroups = 10;

addpath ../

flagDropOut = 0;
flagSparsity = 1;

sparsityHidden = 0.2;

lRate = 0.0005;

diary('out.txt');

wBound = wDim^2;
wLower = -1/wBound;
wUpper = 1/wBound;
W = wLower + 2*wUpper.*rand(wDim,wDim,numGroups);
% W = W/max(max())

l2Norm = 0;

l2RegW = l2Norm;
l2RegU = l2Norm;
l2RegbiasVisOutput = l2Norm;
l2RegbiasHidden = l2Norm;
l2RegbiasVisInput = l2Norm;


uDim = mnistImgDim - wDim + 1;
uBound = uDim^2;
uLower = -1/uBound;
uUpper = 1/uBound;
U = uLower + 2*uUpper.*rand(uDim,uDim,10,numGroups);

biasVisInput = 0;
biasHidden = zeros(uDim, uDim, numGroups);
biasVisOutput = zeros(10,1);

numSamplesInTraining = 5000;
numEpochs = 100;

energyArray = zeros(numEpochs+1,1);
confusionMatArray = zeros(10,10,numEpochs);
accuracyArray = zeros(numEpochs,1);

energyInit = energyConvGroupsBH(trainXImg(:,:,1:numSamplesInTraining),trainY(1:numSamplesInTraining),W,U,biasVisOutput, biasHidden , biasVisInput);
fprintf('Initial Energy = %g\n',energyInit);

energyArray(1) = energyInit;

labels = zeros(10,1);
% W

for i=1:numSamplesInTraining
    trainXImg(:,:,i) = crbm_whiten_olshausen2(trainXImg(:,:,i)) > 0.4;
end
fprintf('Whitened\n');

currEpoch = 1;

if(exist('Saved Variables/data.mat'))
    load('Saved Variables/data.mat');
else
    fprintf('No Saved Variables\n');
end


for epoch=currEpoch:numEpochs
    
    dW = zeros(size(W));
    dU = zeros(size(U));
    dBiasVisInput = 0;
    dBiasVisOutput = zeros(size(biasVisOutput));
    dBiasHidden = zeros(size(biasHidden));
%     count1 = zeros(10,1);
%     count2 = zeros(10,1);
    
    lRate = 0.98*lRate;
    dBiasHiddenSparsity = zeros(size(biasHidden));

    for i=1:numSamplesInTraining

        x1 = trainXImg(:,:,i);
        y1 = zeros(10,1);
        y1(trainY(i)+1) = 1;
        if(epoch == 1)
            labels = labels + y1;
        end
        hid1 = zeros(uDim,uDim,numGroups);
        hidSig = zeros(uDim,uDim,numGroups);
        
        for group = 1:numGroups
            wRot = rot90(W(:,:,group),2);
            convImg = conv2(x1,wRot,'valid');
        %     imshow(convImg)
            hidActivation = convImg + biasHidden(:,:,group) + U(:,:,trainY(i)+1,group);

            if(flagDropOut)
                hidSig(:,:,group) = sigmoid(hidActivation).*(rand(size(convImg)) > 0.5);
            else
                hidSig(:,:,group) = sigmoid(hidActivation);
            end
            
            randMat = rand(size(convImg));
            hid1(:,:,group) = double(hidSig(:,:,group) > randMat);
        end
        %     posAss = conv2(x,rot90(hidSig,2),'valid');
        
        x2Act = zeros(size(x1));
        for group = 1:numGroups
            x2Act = x2Act + conv2(hid1(:,:,group),W(:,:,group));
        end
            x2Act = x2Act + biasVisInput;
            x2Sig = sigmoid(x2Act);
            x2Rand = rand(size(x2Sig));
            x2 = double(x2Sig > x2Rand);

            y2Array = zeros(10,1);
            for group = 1:numGroups
                for j=1:10
                    y2Array(j) = y2Array(j) + trace(hid1(:,:,group)'*U(:,:,j,group));
                end
            end
            y2Array = y2Array + biasVisOutput;
            y2Array = exp(y2Array);
            y2Array = y2Array/sum(y2Array);
            I = discretesample(y2Array,1);
    %         [M,I] = max(y2Array);
            y2 = zeros(10,1);
            y2(I) = 1;


    %         if(epoch == 1 && i==1)
    %             I
    %         end
%             count1(trainY(i)+1) = count1(trainY(i)+1) + 1;
%             count2(I) = count2(I) + 1;
               
            hid2 = zeros(uDim,uDim,numGroups);
            hidSig2 = zeros(uDim,uDim,numGroups);
            
            for group = 1:numGroups
                wRot = rot90(W(:,:,group),2);
                convImg2 = conv2(x2,wRot,'valid');
            %     imshow(convImg)
                hidActivation2 = convImg2 + biasHidden(:,:,group) + U(:,:,trainY(i)+1,group);

                if(flagDropOut)
                    hidSig2(:,:,group) = sigmoid(hidActivation2).*(rand(size(convImg2)) > 0.5);
                else
                    hidSig2(:,:,group) = sigmoid(hidActivation2);
                end
                
                randMat2 = rand(size(convImg2));
                hid2(:,:,group) = double(hidSig2(:,:,group) > randMat2);
            end

    %         dW = dW + (conv2(x1,rot90(hidSig,2),'valid') - conv2(x2,rot90(hidSig2,2),'valid')) / (size(hidSig,1) * size(hidSig,2));
    %         dU(:,:,trainY(i)+1) = dU(:,:,trainY(i)+1) + hidSig;
    %         dU(:,:,I) = dU(:,:,I) - hidSig2;
    %         dBiasVisInput = dBiasVisInput + (sum(sum(x1)) - sum(sum(x2)));
    %         dBiasVisOutput = dBiasVisOutput + (y1-y2);
    %         dBiasHidden = dBiasHidden + (sum(sum(hidSig)) - sum(sum(hidSig2)));
%             for group = 1: numGroups
%                dW(:,:,) = dW + (conv2(x1,rot90(hidSig(:,:,group),2),'valid') - conv2(x2,rot90(hidSig2(:,:,group),2),'valid')) / (uDim^2);
%                dU
%             end
            
            for group = 1: numGroups
            
%                 W(:,:,group) = W(:,:,group) + lRate*((conv2(x1,rot90(hidSig(:,:,group),2),'valid') - conv2(x2,rot90(hidSig2(:,:,group),2),'valid')) / (uDim^2)) - l2RegW*W(:,:,group);
                W(:,:,group) = W(:,:,group) + lRate*((conv2(x1,rot90(hidSig(:,:,group),2),'valid') - conv2(x2,rot90(hidSig2(:,:,group),2),'valid'))) ;
                
                U(:,:,trainY(i)+1,group) = U(:,:,trainY(i)+1,group) + lRate*(hidSig(:,:,group));
                U(:,:,I,group) = U(:,:,I,group) - lRate*(hidSig2(:,:,group));
                
                biasHidden(:,:,group) = biasHidden(:,:,group) + lRate*(hidSig(:,:,group) - hidSig2(:,:,group));
                
%                 biasHidden(:,:,group) = biasHidden(:,:,group) - lRate*((sparsityHidden - hidSig(:,:,group)).*(-hidSig(:,:,group)).*(1 - hidSig(:,:,group)));
                
            end
            biasVisInput = biasVisInput + lRate*(sum(sum(x1)) - sum(sum(x2)));
            biasVisOutput = biasVisOutput + lRate*(y1-y2) - l2RegbiasVisOutput*biasVisOutput;
            
            if(flagSparsity)
                dBiasHiddenSparsity = dBiasHiddenSparsity + (sparsityHidden - hidSig).*(-hidSig).*(1 - hidSig);
            end
%           L2 REGULARIZATION
%             W = W - l2RegW*W;
%             U = U - l2RegU*U;
%             biasHidden = biasHidden - l2RegbiasHidden*biasHidden;
%             biasVisInput = biasVisInput - l2RegbiasVisInput*biasVisInput;
%             biasVisOutput = biasVisOutput - l2RegbiasVisOutput*biasVisOutput;
            
            
    end
    
%     SPARSITY TERM
    if(flagSparsity)
        biasHidden = biasHidden - lRate*(dBiasHiddenSparsity)/numSamplesInTraining;
    end
    if(epoch == 1)
        fprintf('Initial frequency of labels for MNIST dataset \n');
        labels
    end
%     W = W + lRate*dW - l2RegW*W;
%     U = U + lRate*dU - l2RegU*U;
%     biasVisInput = biasVisInput + lRate*dBiasVisInput;
%     biasVisOutput = biasVisOutput + lRate*dBiasVisOutput - l2RegbiasVisOutput*biasVisOutput;
%     biasHidden = biasHidden + lRate*dBiasHidden;

%     biasHidden = biasHidden - 0.005;
    energy = energyConvGroupsBH(trainXImg(:,:,1:numSamplesInTraining),trainY(1:numSamplesInTraining),W,U,biasVisOutput, biasHidden , biasVisInput);
    fprintf('After epoch %d, Energy = %g ',epoch,energy);
    energyArray(epoch+1) = energy;
    
    [accuracy,C,order] = accuracyConvUnSupGroupsBH(testXImg, testY, W, U, biasHidden, biasVisOutput);
    fprintf('Accuracy = %g \n',accuracy);
    fprintf('Confusion Matrix \n');
    C
    accuracyArray(epoch) = accuracy;
    confusionMatArray(:,:,epoch) = C;

    currEpoch = currEpoch + 1;
    save('Saved Variables/data.mat');
    fprintf('Saved epoch %g\n',epoch);
end

% biasVisOutput

diary off;