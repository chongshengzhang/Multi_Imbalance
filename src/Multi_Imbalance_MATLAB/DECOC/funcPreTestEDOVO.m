% Reference:	
% Name: funcPreTestEDOVO.m
% 
% Purpose:  find the prediction array  by all the dichotomy classifiers is closest to which
%           codeword in the ECOC codeword matrix.
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

function prelabel = funcPreTestEDOVO(testdata,code,ft,W,D)
numbertest=size(testdata,1);
numberclass=size(code,1);
testlabel=testdata(1:numbertest,end);

testlabel(:)=0;
testlabel(2)=1;

fX = funcPreEDOVO(testdata,testlabel,ft,D);

% transform the label from 0 to -1, to be consistent with the ECOC codewords;
[a,b]=size(fX);
% for i=1:a
%     for j=1:b
%         if fX(i,j)==0
%             fX(i,j)=-1;
%         end
%     end
% end
fX(fX==0)=-1;


% here uses the imECOC algorithm (decoding for ECOC), for the test data
% reference: Xu-Ying Liu et al. Learning Imbalanced Multi-class Data with Optimal Dichotomy Weights. IEEE ICDM 2013.
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
% In the following, the goal is decoding (using the weight obtained by the function funcwEDOVO()), i.e., find the
% prediction result (array) is closest to which codeword, then output the corresponding original class label.

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

