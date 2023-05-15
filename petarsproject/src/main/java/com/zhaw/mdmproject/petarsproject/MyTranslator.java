package com.zhaw.mdmproject.petarsproject;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class MyTranslator implements Translator<QAInput, String> {
    private List<String> tokens;
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;

    /*
    * This class, MyTranslator, is a custom translator for a Question-Answering (QA) model,
    * implementing the Translator<QAInput, String> interface from the DJL library.
    * It takes a QAInput object with a question and a paragraph and returns the answer as a String.
    */

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        /* This method initializes the vocabulary and tokenizer used for encoding the input
        * the vocabulary is loaded from a text file containing the vocabulary of the model. 
        * The tokenizer is a BERT tokenizer which is used to tokenize the input question and paragraph.
        * Tokenization is the process of splitting the input into tokens, which are the basic units.
        */
        Path path = Paths.get("src/main/resources/bert-base-cased-vocab.txt");
        vocabulary = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
        tokenizer = new BertTokenizer();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) throws IOException {
        /*This method tokenizes and encodes the input question and paragraph into
        NDArrays for indices, attention mask, and token types to be used by the model*/
        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(),
                        input.getParagraph().toLowerCase());

        // get the encoded tokens that would be used in processOutput
        tokens = token.getTokens();
        NDManager manager = ctx.getNDManager();
        // map the tokens(String) to indices(long)
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
        long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();
        NDArray indicesArray = manager.create(indices);
        NDArray attentionMaskArray =
                manager.create(attentionMask);
        NDArray tokenTypeArray = manager.create(tokenType);
        // The order matters
        return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);
    }

    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        /*This method processes the model output to find the start and end indices
        * of the answer in the input paragraph, and converts the tokens back to a String answer. */
        NDArray startLogits = list.get(0);
        NDArray endLogits = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        return tokenizer.tokenToString(tokens.subList(startIdx, endIdx + 1));
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}
