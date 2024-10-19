@echo off

echo Moving models and script back
mv spiritlm\main.py .
mv spiritlm\checkpoints\spiritlm_model Meta_Spirit-LM-ungated
mv spiritlm\checkpoints\speech_tokenizer Meta_Spirit-LM-ungated

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
mv Meta_Spirit-LM-ungated\spiritlm_model spiritlm\checkpoints
mv Meta_Spirit-LM-ungated\speech_tokenizer spiritlm\checkpoints
mv main.py spiritlm

echo.
echo Done! Press the "any" key to close this window.
pause