%% Load in the Fake and Real datasets
filename = "Fake.csv";
dataFake = readtable(filename,'TextType','string');

filename = "Real.csv";
dataReal = readtable(filename,'TextType','string');

%% Title: <Fake and Real news data set (MuMiN-Build)>
% Author: <saattrupdan >
% Date: <24/04/2022>
% Code version: <code version>
% Availability: <https://github.com/MuMiN-dataset/mumin-build>
%% Read the text cells from the excel file
fakeData = dataFake.text;
labelFake = dataFake.label;
fakeData(1:1000) % Paramaters can be changed for deeper analysis. To processing 100 takes around 5 minutes.

realData = dataReal.text;
labelReal = dataReal.label;
realData(1:1000) % Paramaters can be changed for deeper analysis. To processing 100 takes around 5 minutes.
%% %% Preprocess the text

docsFake = preprocessText(fakeData);
docsReal = preprocessText(realData);
%% Place the fake and real processed texts in a bag-of-words-modal
cleanedFake = bagOfWords(docsFake);
cleanedReal = bagOfWords(docsReal);
%% Place the fake and the real bags-of-words in a two seperate word clouds to compare outcomes. 
figure
subplot(1,2,1)
wordcloud(cleanedFake, 'Color','blue');
title("Fake Data Analysis")

subplot(1,2,2)
wordcloud(cleanedReal, 'Color', 'green');
title("Real Data Analysis")
%% Place the data into 6 topics to get a better picture of the topics within the fake news and the real news
rng("default")
numTopics = 6; % These paramaters can be changed for deeper analysis
mdlFake = fitlda(cleanedFake,numTopics,'Verbose',0);

rng("default")
numTopics = 6; % These paramaters can be changed for deeper analysis
mdlReal = fitlda(cleanedReal,numTopics,'Verbose',0);
%% Display the topics of the real and the fake news in multiple word clouds for analysis 
figure
t = tiledlayout("flow");
title(t,"Fake Topics")

for i = 1:numTopics
    nexttile
    wordcloud(mdlFake,i);
    title("Topic " + i)
end

figure
t = tiledlayout("flow");
title(t,"Real Topics")

for i = 1:numTopics
    nexttile
    wordcloud(mdlReal,i);
    title("Topic " + i)
end
%%
strFake = docsFake;
strReal = docsReal;

docsFake;docsReal(1:5) % Paramaters can be changed for deeper analysis. To processing 100 takes around 

fakeDocuments = preprocessText(strFake);
realDocuments = preprocessText(strReal);
%% 
topicMixturesFake = transform(mdlFake, fakeDocuments);
topicMixturesReal = transform(mdlReal, realDocuments);
%% Analyise what the word probability of the Fake data
for i = 1:numTopics
    top = topkwords(mdlFake,3,i);
    topWords(i) = join(top.Word,", ");
end

figure
bar(topicMixturesFake(1,:),'red')

xlabel("Topic")
xticklabels(topWords);
ylabel("Probability")
title("Document Topic Probabilities for Fake News")
%% Do the same analyise for the real data to compare the most popular words
for i = 1:numTopics
    top = topkwords(mdlReal,3,i);
    topWords(i) = join(top.Word,", ");
end

figure
bar(topicMixturesReal(1,:),'green')

xlabel("Topic")
xticklabels(topWords);
ylabel("Probability")
title("Document Topic Probabilities for Real News")

%%Code up to here from Matlab docs

%% Optional clear the data structures and varables
% clear; clc;
%% Second part of the analysis, training the data
% Loading the fake and real news for processing. 

load('real_or_fake_news.mat');
idLabel = find(data.label ~= 'FAKE' & data.label ~= 'REAL');
data(idLabel,:) = [];

idEmpty = strlength(data.text) == 0;
data(idEmpty,:) = [];

data.label = categorical(data.label);

%% Title: <Fake and Real news data set (MuMiN-Build)>
% Author: <saattrupdan >
% Date: <24/04/2022>
% Code version: <code version>
% Availability: <https://github.com/MuMiN-dataset/mumin-build>

%% Visualizing the distribution of the classes

h = histogram(data.label);

%% partitioning the dataset

cvp = cvpartition(data.label,'Holdout',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

%% Partitioning into training and test sets
textDataTrain = dataTrain.text;
textDataTest = dataTest.text;
YTrain = dataTrain.label;
YTest = dataTest.label;

%% %% Preprocessing
documents = preprocessText(textDataTrain);

%% Word Embedding
tic;
embeddingDimension = 100;
embeddingEpochs = 50;

emb = trainWordEmbedding(documents, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'Verbose',1);
toc

%% 
documentLengths = doclength(documents);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

%% Since most of the docs have tokens of size around 1000, padding with seq length of 1000

sequenceLength = 1000;
documentsTruncatedTrain = docfun(@(words) words(1:min(sequenceLength,end)),documents);

%% Converting the doc to sequences to be fed into the LSTM
XTrain = doc2sequence(emb,documentsTruncatedTrain);

%%
for i = 1:numel(XTrain)
    XTrain{i} = leftPad(XTrain{i},sequenceLength);
end
XTrain(1:5);

%% Train Network
inputSize = embeddingDimension;
outputSize = 180;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.005, ...
    'Plots','training-progress', ...
    'Verbose',1);

net = trainNetwork(XTrain,YTrain,layers,options);

%% testing

textDataTest = erasePunctuation(textDataTest);
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);

%% Convert the docs to seq
documentsTruncatedTest = docfun(@(words) words(1:min(sequenceLength,end)),documentsTest);
XTest = doc2sequence(emb,documentsTruncatedTest);
for i=1:numel(XTest)
    XTest{i} = leftPad(XTest{i},sequenceLength);
end
XTest(1:5)

%% Inference
YPred = classify(net,XTest);

%% accuracy
accuracy = sum(YPred == YTest)/numel(YPred);

%%
% Title: <LSTM-fake-news>
% Author: <Peetek>
% Date: <22/01/2018>
% Code version: <code version>
% Availability: <https://github.com/peetak/LSTM-fake-news> 