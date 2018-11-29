% Reference:	
% Name: funcPreE.m
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

function prelabel = funcPreE(testdata,code,ft,W)
numbertest=size(testdata,1);
numberclass=size(code,1);
for t=1:length(ft)
    prec=eval(ft{t},testdata);
    fX(:,t)=cellfun(@str2num, prec);
end
for i=1:numbertest
    ftx=fX(i,:);
    for r=1:numberclass
        for t=1:length(ftx)
            btr(t)=(1-ftx(t)*code(r,t))/2;
        end
        br=btr';
        yall(r)=W*br;
    end

    [minval,minindex]=min(yall);
    prelabel(i)=minindex;
end




