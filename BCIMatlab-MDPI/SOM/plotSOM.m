function plotSOM(weights, xn)
    [ind,dist] = somWinnerNeuron (xn,weights);
    sizef = size(xn,2);
    f = sqrt(sizef);
    if f ~= round(f)
        f = round(f);
        while mod(sizef,f)
            f = f - 1;
        end
    end
    sizeg = size(weights,1);
    g = sqrt(sizeg);
    if g ~= round(g)
        g = round(g);
        while mod(sizeg,g)
            g= g - 1;
        end
    end
    xn = reshape(xn, f, []);
    figure(1)
    subplot(1,2,1)
    pcolor(reshape(dist,g,g))
    subplot(1,2,2)
    image(xn*256)
end