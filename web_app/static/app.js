$(document).ready(function() {
    var selectedFile = null;

    // Show selected file name and store file data
    $('#file').change(function() {
        selectedFile = $(this)[0].files[0];
        if (selectedFile) {
            $('#file-name').text('Selected file: ' + selectedFile.name).show();
        } else {
            $('#file-name').hide();
        }
    });


    // Analyze button click handler
    $('#analyze-btn').click(function() {
        if (!selectedFile) {
            alert('Please choose an image to analyze.');
            return;
        }

        var formData = new FormData();
        formData.append('file', selectedFile);

        // AJAX request to classify endpoint
        $.ajax({
            url: '/classify',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            beforeSend: function() {
                $('#analyze-btn').prop('disabled', true);
                $('#reset-btn').prop('disabled', true);
                $('#analyzing').show();
                $('#result').hide();
            },
            success: function(response) {
                $('#analyze-btn').prop('disabled', false);
                $('#reset-btn').prop('disabled', false);
                $('#analyzing').hide();
                $('#result').show();
                $('#class').text(response.class);
                $('#accuracy').text(response.accuracy.toFixed(2));
            },
            error: function(error) {
                console.log('Error:', error);
                $('#analyze-btn').prop('disabled', false);
                $('#reset-btn').prop('disabled', false);
                $('#analyzing').hide();
                alert('Failed to classify the image. Please try again.');
            }
        });

        // Countdown timer for analyzing (if needed)
        var seconds = 5;
        var interval = setInterval(function() {
            seconds--;
            $('#counter').text(seconds);
            if (seconds == 0) {
                clearInterval(interval);
            }
        }, 1000);
    });

    // Reset button click handler
    $('#reset-btn').click(function() {
        $('#file').val('');
        $('#file-name').hide();
        $('#result').hide();
        selectedFile = null; // Reset selected file
    });
});
