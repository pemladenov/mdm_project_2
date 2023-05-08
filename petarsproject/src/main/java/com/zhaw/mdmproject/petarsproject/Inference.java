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

// This class is used to get the answer from the QA model. It is called from the Controller.java class.
public class Inference {
    //This method uses the input from the user to get the answer from the QA model. 
    public static String qa_predict(QAInput input) throws IOException, TranslateException, ModelException {
        MyTranslator translator = new MyTranslator();
        //The Criteria object is used to load the model and the translator.
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
