function [accuracy,tp_rate,tn_rate,gmean]=cacula(predictionlabel,ftestlabel)
%代码作者：毕璟君 河南大学 2834335964@qq.com
%作者导师：张重生 河南大学 henucs@qq.com
fidx1=(ftestlabel()==1);
p=length(ftestlabel(fidx1));
n=length(ftestlabel(~fidx1)); 
N = p+n;
tp = sum(ftestlabel(fidx1)==predictionlabel(fidx1));
tn = sum(ftestlabel(~fidx1)==predictionlabel(~fidx1));
accuracy = (tp+tn)/N;
tp_rate = tp/p;
tn_rate = tn/n;
gmean = sqrt((tp/p)*(tn/n));
fp = n-tn;
fn = p-tp;
f_measure = 2*tp/(2*tp+fn+fp);
end