function[predFinal] = DS_MCB(member, osKNN, current, pred, data);

newNN = weka.core.Instances(data);
label = current.value(current.numAttributes-1);
vot(1,1:10) = zeros;
limiar = 0.7;

%------------------CÁLCULO DE NOVOS VIZINHOS------------------------------%

if(~wekaPathCheck),wekaCross = []; return,end
    wekaCross = weka.classifiers.Evaluation(osKNN);
    for i=0:4, %pois eu quero 5 vizinhos
        for number = 1 : 10,
            predNN(number) = wekaCross.evaluateModelOnceAndRecordPrediction(member(number), osKNN.get(i)); %avalia cada classificador com os KNN e salva os labels num vetor
        end
        dissimi = sqrt(sum((pred - predNN) .^ 2)); %euclideana entre as predições da amostra atual com as pred de uma vizinha próxima
        simi = 1 - dissimi; %cálculo da similaridade
        if(simi > limiar),
            newNN.add(osKNN.get(i)); %adiciona o vizinho para uma nova seleção de vizinhos
        end
    end
    
    num_newNN = size(newNN); %número de amostras no conjunto newNN
%-------------------------------------------------------------------------%
%--------------------CLASSIFICADOR EXPERT---------------------------------%

    if(~wekaPathCheck),wekaCross = []; return,end
    wekaCross = weka.classifiers.Evaluation(newNN);
    for number = 1 : 10,
        for i=0:num_newNN-1, %num_newNN-1: quantidade total de amostras -1, pois inicia pelo zero
            pred2 = wekaCross.evaluateModelOnceAndRecordPrediction(member(number), newNN.get(i)); %avalia cada classificador com os KNN
            if pred2 == label, vot(number) = vot(number) + 1; end %armazena quantos knn cada classificador acertou
        end
    end

    selection = find(vot > max(vot)-1); %vetor com as POSIÇÕES dos classificadores selecionados (find os máximos)
    [l,c] = size(selection);
    
    trues = 0; falses = 0;
    
    for num_select = 1 : c,
        pred = wekaCross.evaluateModelOnceAndRecordPrediction(member(selection(num_select)), current); %avalia os classificadores selecionados na amostra atual
        if pred == 0, falses = falses + 1;
        else trues = trues + 1;
        end
    end
    
    if (trues >= falses), predFinal = 1;
    else predFinal = 0;
    end
    
    end
    
    