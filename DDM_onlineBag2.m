%MÉTODO ORIGINAL COM DADOS PARCIALMENTE ROTULADOS
function [dataM, change, p, n, emin, smin, classifierReset, member, newmember] = DDM_onlineBag2(data, dataM, current, member, newmember, predex, pred, p_ant, n, emin, smin, classifierReset, allChange);
    numMin_inst = 500; %2500 -- 500(circle)
    %numMin_inst2 = 30;
    alfa = 0.9; %1.5(circle), 1,4(sine) -- 0.9
    beta = 1.05; %1.7(circle), 1.6(sine) -- 1.05
    distP = poissrnd(0.001); %distribuição de Poisson: 0.01(circle,sine) -- 0.001
    
    if (n < numMin_inst), %Se sim, atualiza com true label e sai da função
        
        if pred ~= current.value(current.numAttributes-1),
            p = p_ant + (1 - p_ant)/n; %métrica para o erro prequential
        else p = p_ant - p_ant/n; %acerto
        end
        
        while (distP > 0), %O membro atuali com 'distP' cópias do exemplo 'current'
            member.updateClassifier(current); %classificador aprende incrementalmente
            distP = distP - 1;
        end
            level = 'InControl';
            change = 0;
            n = n + 1;
            return; %passa para a próxima iteração (amostra)
    end
    % --------------------------------------------------------------------%
    %Só segue para os comandos abaixo após n >= 30
    
    if pred ~= predex,
         p = p_ant + (1 - p_ant)/n; %métrica para o erro prequential
    else p = p_ant - p_ant/n; %acerto
    end

    s = sqrt(p*(1-p)/n); %desvio padrão
    n = n+1;
        
    if (p + s <= emin + smin),
        emin = p; 
        smin = s;
    end
    
      if (allChange >= 1), %HORA DE RESETAR
            member = newmember;
            %fprintf(size(dataM));
            newmember = trainWekaClassifier(data,'trees.HoeffdingTree');

            %Resetando---------------------------------------%
            p = 1;
            s = 0;
            n = 2; %DIFERENTE, DEVERIA SETAR PARA 1
            emin = 1;
            smin = 1;
            change = 0;
            %------------------------------------------------%
      end
    
    
    if (p + s > emin + beta * smin & size(dataM)>=5),% & n > 2*numMin_inst),
                level = 'Drift'; change = 1;
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
                else if (p + s > emin + alfa * smin),
                        level = 'Warning';
                        if (classifierReset == 1), %Primeira vez no warning após CONTROL LEVEL
                            newmember = trainWekaClassifier(data,'trees.HoeffdingTree'); %Reseta o classificador reserva
                            dataM = weka.core.Instances(data); %zera o conjunto de validação
                            classifierReset = 0;
                        end
                        dataM.add(current); %adiciona amostra no conjunto de validação com o rótulo VERDADEIRO
                        %
                        current.setValue(current.numAttributes-1, predex); %aqui seta o suposto rótulo também para as amostras do warning level
                        %dataM.add(current); %adiciona amostra no conjunto de validação com o rótulo PSEUDO
                        %
                        while (distP > 0), %atualiza o classificador reserva com distP cópias
                            newmember.updateClassifier(current); %Quando cai no Warning, sempre atualiza com o suposto rótulo
                            distP = distP - 1;
                        end
                        change = 0;
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %                    
                    else level = 'InControl';
                         classifierReset = 1;
                         change = 0;
                end
    end
    
       %if (totVote == 1),
           current.setValue(current.numAttributes-1, predex); %Assume a predição da maioria como TRUE LABEL e passará a atualizar com este suposto label
       %end
        while (distP > 0), %O membro atualiza com 'distP' cópias do exemplo 'current'
            member.updateClassifier(current); %classificador aprende incrementalmente
            distP = distP - 1;
        end
       %end    
end