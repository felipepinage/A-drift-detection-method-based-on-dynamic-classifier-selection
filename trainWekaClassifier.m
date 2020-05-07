function wekaClassifier = trainWekaClassifier(wekaData,type,options)
% Train a weka classifier.
%
% wekaData - A weka java Instances object holding all of the training data.
%            You can convert matlab data to this format via the
%            matlab2weka() function or load existing weka arff data using
%            the loadARFF() function. 
%
% type    -  A string naming the type of classifier to train relative to
%            the weka.classifiers package. There are many options - see
%            below for a few. See the weka documentation for the rest. 
%
% options - an optional cell array of strings listing the options specific
%           to the classifier. See the weka documentation for details. 
%
% Example: 
% wekaClassifier = trainWekaClassifier(data,'bayes.NaiveBayes',{'-D'});
%
% List of a few selected weka classifiers - there are many many more:
% 
% bayes.BayesNet
% bayes.NaiveBayes
% bayes.NaiveBayesMultinomial
% bayes.HNB
% functions.GaussianProcesses
% functions.IsotonicRegression
% functions.Logistic
% functions.MultilayerPerceptron
% functions.RBFNetwork
% functions.SVMreg
% lazy.IBk
% lazy.LBR
% misc.HyperPipes
% trees.RandomForest
% ...

    if(~wekaPathCheck),wekaClassifier = []; return,end
    wekaClassifier = javaObject(['weka.classifiers.',type]); %Criando um javaObject chamando o construtor weka.classifiers.xxx
    if(nargin == 3 && ~isempty(options))
        wekaClassifier.setOptions(options); %Como já chamei o construtor, já posso usar suas subfunções
    end
    wekaClassifier.buildClassifier(wekaData);
end

% multC = weka.classifiers.meta.Bagging;
% multC.setOptions({'-P' '50', '-W' 'weka.classifiers.trees.J48'});
% multC.buildClassifier(trainData);

