$(document).ready(function () {
    $('#qa-form').on('submit', function (event) {
        event.preventDefault();
        let question = $('#question').val();
        let paragraph = $('#paragraph').val();
        if (question && paragraph) {
            $.ajax({
                type: 'POST',
                url: '/ask',
                data: {
                    question: question,
                    paragraph: paragraph
                },
                success: function (response) {
                    $('#answer').text(response);
                    $('#answer-container').show();
                },
                error: function () {
                    alert('Error occurred while processing the request');
                }
            });
        } else {
            alert('Please fill out both the question and paragraph fields');
        }
    });
});
