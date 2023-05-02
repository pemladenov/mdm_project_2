package com.zhaw.mdmproject.petarsproject;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import ai.djl.modality.nlp.qa.QAInput;

@RestController
public class Controller {

    @PostMapping("/ask")
    public String ask(@RequestParam("question") String question, @RequestParam("paragraph") String paragraph) {
        QAInput input = new QAInput(question, paragraph);
        String answer = "";
        try {
            answer = Inference.qa_predict(input);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return answer;
    }
}
