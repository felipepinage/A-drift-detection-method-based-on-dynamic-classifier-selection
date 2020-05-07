%CRIA MEMBROS COM PARAMETROS DIFERENTES, MAS SEMPRE APRENDENDO COM AS
%MESMAS AMOSTRAS (no in�cio).

clear;clc;
javaaddpath('weka.jar');

%Constr�i O CLASSIFICADOR
[data, loader] = incrementalARFF('kddcup99.arff'); %Retorna a estrutura do conjunto (nome e atributos), mas sem dados (vazio)
[data2] = loadARFF('kddcup99_valid.arff'); %Base de valida��o inicial

TAM_CJ = 10;

for number = 1:TAM_CJ,
    dataM(number) = weka.core.Instances(data); %Um vetor para armazenar conjuntos de dados para valida��o
    member(number) = trainWekaClassifier(data,'trees.HoeffdingTree'); %Constr�i o classificador
    newmember(number) = trainWekaClassifier(data,'trees.HoeffdingTree'); %Constr�i o classificador reserva
end


%INICIA O PROCESSO PARA TESTE
current = loader.getNextInstance(data); %Recebe a instancia atual do conjunto de dados

%INICIALIZA��ES
labeled = 0; %n�mero total de amostras rotuladas
wrong = 0; right = 0; wrongD = 0; rightD = 0; %para calcular acur�cia total.
n(1,1:TAM_CJ) = 1; nMJ = 1; %n�mero de amostras por conceito
i = 1; %conta o numero de amostras
pMJ(i) = 1; p(1:TAM_CJ,1) = 1; %erro prequencial
emin(1,1:TAM_CJ) = 1; %erro prequencial m�nimo para DDM
smin(1,1:TAM_CJ) = 1; %desvio padr�o m�nimo para DDM
cReset(1,1:TAM_CJ) = 0; %0: classif reserva est� criado; 1: criar novo classif reserva
allChange = 0; napc = 1; i_parou = 1;
assLabel = 0;
soma = 0; %n�mero de detec��es
ep(1:TAM_CJ,1) = 0; %distancia media de erros para o EDDM
p2smax(1,1:TAM_CJ) = 0; %maximo para o EDDM
d(1,1:TAM_CJ) = 0; %inicio/fim distancia para EDDM
s_temp(1,1:TAM_CJ) = 0; %desvio padrao tempor�rio para c�lculo do desvio padrao atual (EDDM)
num_erros(1,1:TAM_CJ) = 0; %conta n�mero de erros para cada membro no EDDM

tic;
while (isempty(current) == 0) %Enquanto houver instancia (amostra)
    
    [pred, majVote, totVote] = predBagging_v3(member, TAM_CJ, data, current); %pred: vetor de predi��es individuais; majVote: voto majorit�rio; totVote: se todos concordam.
     
    wekaNN = weka.core.neighboursearch.LinearNNSearch(data2);
    osKNN = wekaNN.kNearestNeighbours(current, 5); %Encontra os 5 vizinhos mais pr�ximos de current no conjunto de valida��o
    
    [predex] = DSC_LA(member, osKNN, current); %(DSC-LA)predi��o dos experts: assumirei como true label
    %[predex] = DS_MCB(member, osKNN, current, pred, data); %(DS-MCB) m�todo baseado em classifier behavior
    
    
    if (totVote == 1), assLabel = assLabel + 1; end
    
    i = i+1; %contador para erros prequenciais
    
    clc; i
    

    if current.value(current.numAttributes-1) ~= predex,
         pMJ(i) = pMJ(i-1) + (1 - pMJ(i-1))/nMJ;
         wrong = wrong + 1;
    else pMJ(i) = pMJ(i-1) - pMJ(i-1)/nMJ;
         right = right + 1;
    end 
    nMJ = nMJ + 1; %num de amostras por conceito para avalia��o do conjunto
    
    for num = 1 : TAM_CJ,
        [dataM(num), change(num), p(num, i), n(num), emin(num), smin(num), cReset(num), member(num), newmember(num)] = DDM_onlineBag2(data, dataM(num), current, member(num), newmember(num), predex, pred(num), p(num, i-1), n(num), emin(num), smin(num), cReset(num), allChange);
        %[dataM(num), change(num), n(num), ep(num, i), d(num), s_temp(num), num_erros(num) p2smax(num), cReset(num), member(num), newmember(num)] = EDDM_onlineBag(data, dataM(num), current, member(num), newmember(num), predex, pred(num), n(num), ep(num, i-1), d(num), s_temp(num), num_erros(num), p2smax(num), cReset(num), allChange);
        if change(num) == 1, data2 = dataM(num); end %labeled = labeled + size(dataM(num)); end
    end
    allChange = sum(change); %numero de membros que detectaram drift
    
    if allChange >= 1, nMJ = 1; pMJ(i) = 1; soma = soma + 1; labeled = labeled + size(data2); end
    
    current = loader.getNextInstance(data); %L� amostras uma por uma
    
    %if i == 100000,
    %    return;
    %end
end
tempo = toc;