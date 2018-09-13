function result=similarity(valuesX,valuesY)

global aMaxs aMins;

sim=0;
natt=length(valuesX);
for a=1:natt
    %%// if not a NOMINAL Attribute (but a numeric attribute)
    if 1
        aMaxs(a) = max(aMaxs(a), valuesY(a)); % y is the unseen test instance
        aMins(a) = min(aMins(a), valuesY(a));
        denom = aMaxs(a) - aMins(a);
        if denom > 0
            sim =sim+ 1.0 - abs(valuesX(a) - valuesY(a))/denom;
        else
            %elements necessarily have the same values
            sim=sim+1;
        end
    else
        if valuesX(a)==valuesY(a)
            sim=sim+1.0;
        else
            sim=sim+0.0;
        end
    end
end
result=sim / natt;
end
