function [ e ] = energyConvGroupsBH( dataX,dataY,W,U,biasVisOutput, biasHidden , biasVisInput)
e=0;

for i=1:size(dataX,3)
    img = dataX(:,:,i);
    y = zeros(10,1);
    y(dataY(i)+1) = 1;
    numGroups = size(W,3);
    
    for group=1:numGroups
        
        wRot = rot90(W(:,:,group),2);
        convImg = conv2(img,wRot,'valid');
%     imshow(convImg)
        hidActivation = convImg + biasHidden(:,:,group) + U(:,:,dataY(i)+1,group);
        hidSig = sigmoid(hidActivation);
        randMat = rand(size(convImg));
        hid1 = double(hidSig > randMat);
        
        term1 = trace(hid1'*convImg);
        term3 = trace(biasHidden(:,:,group)'*hid1);
        term5 = trace(hid1'*U(:,:,dataY(i)+1));
        e = e - term1 - term3 - term5;
    end
    term2 = biasVisInput*sum(sum(img));
    term4 = biasVisOutput(dataY(i)+1);
    e = e - term2 - term4;
    
end

end