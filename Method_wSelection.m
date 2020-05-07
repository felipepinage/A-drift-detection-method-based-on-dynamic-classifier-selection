%CRIA MEMBROS COM PARAMETROS DIFERENTES, MAS SEMPRE APRENDENDO COM AS
%MESMAS AMOSTRAS (no início).

clear;clc;
javaaddpath('weka.jar');

%Constrói O CLASSIFICADOR
[data, loader] = incrementalARFF('kddcup99.arff'); %Retorna a estrutura do conjunto (nome e atributos), mas sem dados (vazio)
[data2] = loadARFF('kddcup99_valid.arff'); %Base de validação inicial

TAM_CJ = 10;

for number = 1:TAM_CJ,
    dataM(number) = weka.core.Instances(data); %Um vetor para armazenar conjuntos de dados para validação
    member(number) = trainWekaClassifier(data,'trees.HoeffdingTree'); %Constrói o classificador
    newmember(number) = trainWekaClassifier(data,'trees.HoeffdingTree'); %Constrói o classificador reserva
end


%INICIA O PROCESSO PARA TESTE
current = loader.getNextInstance(data); %Recebe a instancia atual do conjunto de dados

%INICIALIZAÇÕES
labeled = 0; %número total de amostras rotuladas
wrong = 0; right = 0; wrongD = 0; rightD = 0; %para calcular acurácia total.
n(1,1:TAM_CJ) = 1; nMJ = 1; %número de amostras por conceito
i = 1; %conta o numero de amostras
pMJ(i) = 1; p(1:TAM_CJ,1) = 1; %erro prequencial
emin(1,1:TAM_CJ) = 1; %erro prequencial mínimo para DDM
smin(1,1:TAM_CJ) = 1; %desvio padrão mínimo para DDM
cReset(1,1:TAM_CJ) = 0; %0: classif reserva está criado; 1: criar novo classif reserva
allChange = 0; napc = 1; i_parou = 1;
assLabel = 0;
soma = 0; %número de detecções
ep(1:TAM_CJ,1) = 0; %distancia media de erros para o EDDM
p2smax(1,1:TAM_CJ) = 0; %maximo para o EDDM
d(1,1:TAM_CJ) = 0; %inicio/fim distancia para EDDM
s_temp(1,1:TAM_CJ) = 0; %desvio padrao temporário para cálculo do desvio padrao atual (EDDM)
num_erros(1,1:TAM_CJ) = 0; %conta número de erros para cada membro no EDDM

tic;
while (isempty(current) == 0) %Enquanto houver instancia (amostra)
    
    [pred, majVote, totVote] = predBagging_v3(member, TAM_CJ, data, current); %pred: vetor de predições individuais; majVote: voto majoritário; totVote: se todos concordam.
     
    wekaNN = weka.core.neighboursearch.LinearNNSearch(data2);
    osKNN = wekaNN.kNearestNeighbours(current, 5); %Encontra os 5 vizinhos mais próximos de current no conjunto de validação
    
    [predex] = DSC_LA(member, osKNN, current); %(DSC-LA)predição dos experts: assumirei como true label
    %[predex] = DS_MCB(member, osKNN, current, pred, data); %(DS-MCB) método baseado em classifier behavior
    
    
    if (totVote == 1), assLabel = assLabel + 1; end
    
    i = i+1; %contador para erros prequenciais
    
    clc; i
    

    if current.value(current.numAttributes-1) ~= predex,
         pMJ(i) = pMJ(i-1) + (1 - pMJ(i-1))/nMJ;
         wrong = wrong + 1;
    else pMJ(i) = pMJ(i-1) - pMJ(i-1)/nMJ;
         right = right + 1;
    end 
    nMJ = nMJ + 1; %num de amostras por conceito para avaliação do conjunto
    
    for num = 1 : TAM_CJ,
        [dataM(num), change(num), p(num, i), n(num), emin(num), smin(num), cReset(num), member(num), newmember(num)] = DDM_onlineBag2(data, dataM(num), current, member(num), newmember(num), predex, pred(num), p(num, i-1), n(num), emin(num), smin(num), cReset(num), allChange);
        %[dataM(num), change(num), n(num), ep(num, i), d(num), s_temp(num), num_erros(num) p2smax(num), cReset(num), member(num), newmember(num)] = EDDM_onlineBag(data, dataM(num), current, member(num), newmember(num), predex, pred(num), n(num), ep(num, i-1), d(num), s_temp(num), num_erros(num), p2smax(num), cReset(num), allChange);
        if change(num) == 1, data2 = dataM(num); end %labeled = labeled + size(dataM(num)); end
    end
    allChange = sum(change); %numero de membros que detectaram drift
    
    if allChange >= 1, nMJ = 1; pMJ(i) = 1; soma = soma + 1; labeled = labeled + size(data2); end
    
    current = loader.getNextInstance(data); %Lê amostras uma por uma
    
    %if i == 100000,
    %    return;
    %end
end
tempo = toc;