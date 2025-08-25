# Report:

## Initial findings:
- After plotting the audio data against time, it was found that there is on the actual audio a large amount of noise at all the duration of the audio file.
- After plotting the audio data against frequency, it was found that the noise was distributed through all frequencies from 0 till 8000 Hz, which indicates there is a mask to the original audio with a starting value just below 1000 Hz(around 750 Hz).
- From the spectrogram, two bright lines appear(one just above 2000 Hz and the other around 1300 Hz) with high magnitude.

## Data transformation:
- Knowing the normal human speech frequency is between 0 to 4000 Hz, so we will perform a low pass filter to remove any frequencies higher than 4000 Hz.
- Also, a band-stop filter will be applied twice, one for each frequency, for the frequencies that have high magnitude that may be masking the audio to remove them.

## Password: 315
### Finally, the password was found, but unfortunatly the audio lost some of its volume making it baerly hearable.