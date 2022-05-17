function C = doc2sequence(emb,documents)

parfor i = 1:numel(documents)
    words = string(documents(i));
    idx = ~isVocabularyWord(emb,words);
    words(idx) = [];
    C{i} = word2vec(emb,words)';
end

end