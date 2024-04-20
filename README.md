# IIT-Bombay-Machine-Learning-Intern

Link: https://github.com/openai/whisper
Link: https://github.com/AI4Bharat/vistaar
Link: https://github.com/belambert/asr-evaluation
Test Dataset Link: https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_3h.tar.gz
      
      Hello,
      
      Are you still looking for the internship at IIT Bombay for the Machine Learning Profile?
      If Yes then Kindly do this task. we expect you to complete a short assignment that just takes one day. But we are providing 7 days to complete.  
      
      You must implement the Whisper Hindi (Automatic Speech Recognition) ASR model. Also calculate the WER for Kathbath dataset
      Just give us WER for the kathbath dataset.
      
      Thanking You
      Ayush
      IIT Bombay

# Word Error Rate (WER)

Implementing an Automatic Speech Recognition (ASR) model like Whisper Hindi would typically involve several steps:

    Data Collection and Preprocessing: Gather a large dataset of Hindi speech recordings. Preprocess the data by converting audio files into a format suitable for training a neural network.

    Model Architecture Selection: Choose a suitable architecture for your ASR model. Common choices include convolutional neural networks (CNNs), recurrent neural networks (RNNs) such as long short-term memory (LSTM) or gated recurrent unit (GRU), and transformer-based models like BERT or wav2vec.

    Training: Train the ASR model using the collected and preprocessed data. This involves feeding the audio data into the model and adjusting its parameters (weights) based on the error between the predicted transcriptions and the ground truth transcriptions.

    Evaluation: Evaluate the performance of the trained model on a separate dataset using metrics such as Word Error Rate (WER), Character Error Rate (CER), or Sentence Error Rate (SER). WER is commonly used in ASR tasks and measures the proportion of words in the predicted transcription that differ from the reference transcription, normalized by the total number of words in the reference transcription.

Here's a simplified code snippet for calculating the Word Error Rate (WER) using the Levenshtein distance algorithm:

python

                  def wer(reference, hypothesis):
                      """
                      Calculate Word Error Rate (WER) between reference and hypothesis.
                      """
                      reference = reference.split()
                      hypothesis = hypothesis.split()
                  
                      # Create a matrix of size (len(reference) + 1) x (len(hypothesis) + 1)
                      matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
                  
                      # Initialize the first row and column of the matrix
                      for i in range(len(reference) + 1):
                          matrix[i][0] = i
                      for j in range(len(hypothesis) + 1):
                          matrix[0][j] = j
                  
                      # Fill in the matrix
                      for i in range(1, len(reference) + 1):
                          for j in range(1, len(hypothesis) + 1):
                              if reference[i - 1] == hypothesis[j - 1]:
                                  matrix[i][j] = matrix[i - 1][j - 1]
                              else:
                                  matrix[i][j] = min(matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1]) + 1
                  
                      # Return WER
                      return float(matrix[len(reference)][len(hypothesis)]) / len(reference)
                  
                  # Example usage:
                  reference_transcription = "मैं आज बाजार गया"
                  hypothesis_transcription = "मैं आज बाजार जाया"
                  wer_score = wer(reference_transcription, hypothesis_transcription)
                  print("WER:", wer_score)

This code calculates the WER between a reference transcription and a hypothesis transcription by first splitting them into lists of words and then using the Levenshtein distance algorithm to compute the minimum number of edits (insertions, deletions, substitutions) required to transform the hypothesis into the reference transcription. Finally, it normalizes this edit distance by the length of the reference transcription to obtain the WER.

Python module for evaluting ASR hypotheses (i.e. word error rate and word recognition rate).

This module depends on the editdistance project, for computing edit distances between arbitrary sequences.

The formatting of the output of this program is very loosely based around the same idea as the align.c program commonly used within the Sphinx ASR community. This may run a bit faster if neither instances nor confusions are printed.

Please let me know if you have any comments, questions, or problems.
Output

The program outputs three standard measurements:

    Word error rate (WER)
    Word recognition rate (the number of matched words in the alignment divided by the number of words in the reference).
    Sentence error rate (SER) (the number of incorrect sentences divided by the total number of sentences).

