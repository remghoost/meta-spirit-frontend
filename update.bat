@echo off

echo Moving models and script back
move spiritlm\main.py .
move spiritlm\checkpoints\spiritlm_model Meta_Spirit-LM-ungated
move spiritlm\checkpoints\speech_tokenizer Meta_Spirit-LM-ungated

echo. 
echo Updating main repo
git pull

echo.
echo Updating spiritlm
cd spiritlm
git pull
cd ..

echo.
echo Moving things back
move Meta_Spirit-LM-ungated\spiritlm_model spiritlm\checkpoints
move Meta_Spirit-LM-ungated\speech_tokenizer spiritlm\checkpoints
move main.py spiritlm

echo.
echo Done! Press the "any" key to close this window.
pause