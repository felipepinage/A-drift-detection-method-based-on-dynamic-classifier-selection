%M�TODO ORIGINAL COM DADOS PARCIALMENTE ROTULADOS
function [dataM, change, n, ep, d, s_temp, num_erros, p2smax, classifierReset, member, newmember] = EDDM_onlineBag(data, dataM, current, member, newmember, predex, pred, n, ep_ant, d, s_temp, num_erros, p2smax, classifierReset, allChange);
    %INICIALIZANDO
    change = 0;

    numMin_inst = 30; %numero m�nimo de erros
    alfa = 0.9; %0.9
    beta = 0.85; %0.85
    distP = poissrnd(0.001); %distribui��o de Poisson %tava 0.08
    
    if (allChange >= 1), %HORA DE RESETAR: algum membro na itera��o anterior detectou drift
            member = newmember;
            newmember = trainWekaClassifier(data,'trees.HoeffdingTree');

            %Resetando---------------------------------------%
            ep = 0;
            s = 0;
            n = 1; %DIFERENTE, DEVERIA SETAR PARA 1
            p2smax = 0;
            change = 0;
            num_erros = 0; %Conta o numero de erros
            last_d = 0; %Inicio da distancia
            d = 0; %Fim da distancia
            s_temp = 0;
            %------------------------------------------------%
    end
    
%-----------------------------------------------------------------------------------------------------%
%-----------------------------------------------------------------------------------------------------%
    n = n+1;
    if (n < numMin_inst), %Se sim, atualiza com true label e sai da fun��o
         rotulo = current.value(current.numAttributes-1);
    else rotulo = predex;
    end
        
    if pred ~= rotulo, %erro
        num_erros = num_erros + 1;
        last_d = d;
        d = n - 1;
        dist = d - last_d;
        old_ep = ep_ant;
        ep = ep_ant + (dist - ep_ant)/num_erros; %distancia de erros
        s_temp = s_temp + (dist - ep)*(dist - old_ep);
        s = sqrt(s_temp/num_erros);
        p2s = ep+2*s;
        
        %S� segue para os comandos abaixo ap�s n >= 30
    
    if (p2s > p2smax), %ainda na condi��o de erro
        if (n > numMin_inst)
            p2smax = p2s; %atualiza p2smax
        end
        
    else div = p2s/p2smax;
         if (div < beta & dataM.numInstances() >= 5), %detectou drift
                level = 'Drift'; change = 1;
                
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %

         else if (div < alfa & num_erros >= numMin_inst), %caiu no warning
                level = 'Warning';
                if (classifierReset == 1), %Primeira vez no warning ap�s CONTROL LEVEL
                    newmember = trainWekaClassifier(data,'trees.HoeffdingTree'); %Reseta o classificador reserva
                    dataM = weka.core.Instances(data); %zera o conjunto de valida��o
                    classifierReset = 0;
                end
                dataM.add(current); %adiciona amostra no conjunto de valida��o com o r�tulo VERDADEIRO
                current.setValue(current.numAttributes-1, predex); %aqui seta o suposto r�tulo tamb�m para as amostras do warning level
                %dataM.add(current); %adiciona amostra no conjunto de valida��o com o r�tulo PSEUDO
                %
                while (distP > 0), %atualiza o classificador reserva com distP c�pias
                    newmember.updateClassifier(current); %Quando cai no Warning, sempre atualiza com o suposto r�tulo
                    distP = distP - 1;
                end
                change = 0;
                
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %

              else level = 'InControl'; %errou mas continua est�vel
                   classifierReset = 1;
                   change = 0;
             end
         end
    end %end erros
        
    else ep = ep_ant; %ACERTO
    end

    
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
    
    current.setValue(current.numAttributes-1, rotulo); %Assume a predi��o da maioria como TRUE LABEL e passar� a atualizar com este suposto label
    
    while (distP > 0), %O membro atualiza com 'distP' c�pias do exemplo 'current'
       member.updateClassifier(current); %classificador aprende incrementalmente
       distP = distP - 1;
    end
  
end