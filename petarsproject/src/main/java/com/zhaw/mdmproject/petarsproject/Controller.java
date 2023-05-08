package com.zhaw.mdmproject.petarsproject;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import ai.djl.modality.nlp.qa.QAInput;

@RestController
public class Controller {

    // This is the endpoint for the QA model
    @PostMapping("/ask")
    public String ask(@RequestParam("question") String question, @RequestParam("paragraph") String paragraph) {
        // Creating a new QAInput object with the question and paragraph from the user
        QAInput input = new QAInput(question, paragraph);
        String answer = "";
        // Calling method from Inference.java to get the answer with the input from the user
        try {
            answer = Inference.qa_predict(input);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return answer;
    }

    // For testing purposes to check if the server is running
    @GetMapping("/ping")
    public String ping() {
        return "pong";
    }
}
