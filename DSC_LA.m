function[predFinal] = DSC_LA(member, osKNN, current);

label = current.value(current.numAttributes-1);
vot(1,1:10) = zeros;

if(~wekaPathCheck),wekaCross = []; return,end
    wekaCross = weka.classifiers.Evaluation(osKNN);
    for number = 1 : 10,
        for i=0:4, %pois eu quero 5 vizinhos
            pred = wekaCross.evaluateModelOnceAndRecordPrediction(member(number), osKNN.get(i)); %avalia cada classificador com os KNN
            if pred == label, vot(number) = vot(number) + 1; end %armazena quantos knn cada classificador acertou
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
    
    