function [accuracy, C, order] = accuracyConvUnSupGroupsBH(testX, testY, W, U, biasHidden, biasVisOutput)
   
    numTestSamples = 10000;
%     numTestSamples = size(testX,3);
    numGroups = size(W,3);
    numCorrect = 0;
    actualLabels = testY(1:numTestSamples) + 1; 
    predictedLabels = zeros(numTestSamples,1);
    
%     biasVisOutput
    
    for i=1:numTestSamples
%     for i=1:1
       x = testX(:,:,i);
       convTerm = zeros(size(U,1),size(U,2),numGroups);
       for group = 1:numGroups
            convTerm(:,:,group) = conv2(x,rot90(W(:,:,group),2),'valid');
       end
       
       yProb = zeros(10,1);
       for j=1:10
           activationTerm = zeros(size(convTerm));
           for group = 1:numGroups
                activationTerm(:,:,group) = biasHidden(:,:,group) + U(:,:,j,group) + convTerm(:,:,group);
           end
           expTerm = exp(activationTerm);
           sumTerm = 1 + expTerm;
           
           % PRODUCT
%            productTerm = prod(sumTerm);
%            yProb(j) = productTerm*exp(biasVisOutput(j));
           
           % LOG 
           logTerms = log(sumTerm);
           sumLogTerms = sum(sum(sum(logTerms)));
           yProb(j) = sumLogTerms + biasVisOutput(j);
       end
%        yProb
       [M,I] = max(yProb);
       predictedLabels(i) = I;
       if(actualLabels(i) == I)
           numCorrect = numCorrect + 1;
       end
    end
    [C, order] = confusionmat(actualLabels, predictedLabels);
    accuracy = numCorrect/numTestSamples*100.0;
    
end