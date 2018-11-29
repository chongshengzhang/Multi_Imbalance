% Reference:	
% Name: similarity.m
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
