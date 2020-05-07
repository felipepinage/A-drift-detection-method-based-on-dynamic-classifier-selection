function [pred, majVote, totVote] = predBagging_v3(member, TAM_CJ, data, current)
    %import weka.classifiers.Evaluation;
    
    trues = 0; falses = 0;
    
    if(~wekaPathCheck),wekaCross = []; return,end
    wekaCross = weka.classifiers.Evaluation(data);
    
    for number = 1 : TAM_CJ,
        pred(number) = wekaCross.evaluateModelOnceAndRecordPrediction(member(number), current); %avalia o classificador 1 com a amostra atual
        if (pred(number) == 1), trues = trues + 1;
        else falses = falses + 1; end
    end
    
    n = 6; %numero desejado de classificadores que concordam
    
    if (trues > falses), majVote = 1;
    else majVote = 0; end
        
    if (trues >= n | falses >= n),
        totVote = 1; %todos os membros votaram na mesma classe.
    else totVote = 0;
            
end