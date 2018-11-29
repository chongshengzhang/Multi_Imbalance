function target_celle = crea_array_celle_etichette(target)

%  funzione che crea target di celle a partire da target double: in
%  particolare mette una stringa in corrispondenza di ogni classe


target_celle = cell(size(target));


for i = 1: length(target)
    target_celle{i} = ['classe', num2str(target(i))] ; 
end
    

end