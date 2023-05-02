package com.zhaw.mdmproject.petarsproject;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;


public class Inference {
     /* public static void main(String[] args) throws IOException, TranslateException, ModelException {
      String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                + "Which operated between December 2004 and April 2006. "
                + "It ceased operations after its Japanese distributor folded.";
        QAInput input = new QAInput(question, paragraph);
        
        String answer = Inference.qa_predict(input);
        System.out.println("The answer is: \n" + answer);
    } */

    public static String qa_predict(QAInput input) throws IOException, TranslateException, ModelException {
        MyTranslator translator = new MyTranslator();
        Criteria<QAInput, String> criteria = Criteria.builder()
                .setTypes(QAInput.class, String.class)
                .optModelPath(Paths.get("petarsproject/src/main/resources/trace_cased_bertqa.pt"))
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        ZooModel<QAInput, String> model = criteria.loadModel();
        try (Predictor<QAInput, String> predictor = model.newPredictor(translator)) {
            return predictor.predict(input);
        }
    }
}
