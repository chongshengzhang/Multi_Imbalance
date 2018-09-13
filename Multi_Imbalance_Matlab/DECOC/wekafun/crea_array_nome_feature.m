function [array_celle] = crea_array_nome_feature(numero_ripetizioni)

% Funzione che crea un array di celle cos?fatto:
% {'Feature1', 'Feature2', ...,'FeatureN', 'class'}

% if numero_ripetizioni == 1
%     error('Errore 1')
% end


for i = 1: numero_ripetizioni
    
    if i == 1
        array_celle{i} = {'Feature1'};
    else
        array_celle = cat(2, array_celle, {['Feature', num2str(i)]});
    end
    
end 

array_celle = cat(2, array_celle, {'class'});


end