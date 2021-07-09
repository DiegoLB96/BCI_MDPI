function lP = backward(lP, err, numPatterns)

numLayers = size(lP,2);

% Gradient Calculation
delta = err .* (1+lP(end).activation).*(1-lP(end).activation);
for i = numLayers-1:-1:1
    ss = -lP(1).LR.*lP(i).activation*delta' / numPatterns;
    if i == numLayers-1
        lP(i).weightsinc = ss(1:end-1,:) + lP(1).Mom.*lP(i).weightsinc;
        lP(i).biasinc = ss(end,:) + lP(1).Mom.*lP(i).biasinc;
        %  delta = (1+layerProp(i).activation) .* (1-layerProp(i).activation) .* ([layerProp(i).weights' layerProp(i).bias']'*delta);
        %  delta = delta';
    else
        lP(i).weightsinc = ss(1:end-1,:) + lP(1).Mom.*lP(i).weightsinc;
        lP(i).biasinc = ss(end,:) + lP(1).Mom.*lP(i).biasinc;
    end
    if i > 1
        delta = (1+lP(i).activation) .* (1-lP(i).activation) .* ([lP(i).weights' lP(i).bias']'*delta);
    end
end

% update weights
for i = 1:numLayers-1
    lP(i).weights = lP(i).weights + lP(i).weightsinc;
    lP(i).bias = lP(i).bias + lP(i).biasinc;
end

end