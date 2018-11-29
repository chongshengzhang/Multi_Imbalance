% Reference:	
% Name: cacula.m
% 
% Authors: Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% Copyright: (c) 2018 Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% This file is a part of Multi_Imbalance software, a software package for multi-class Imbalance learning. 
% 
% Multi_Imbalance software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
% as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
% Multi_Imbalance software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program. 
% If not, see <http://www.gnu.org/licenses/>.

function [accuracy,tp_rate,tn_rate,gmean]=cacula(predictionlabel,ftestlabel)

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