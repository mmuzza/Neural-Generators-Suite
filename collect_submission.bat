@echo off
del assignment_submission.zip 2>nul
powershell -Command "Compress-Archive -Path configs, models\*.py, losses\*.py, utils\*.py, outputs\vae\*.pth, outputs\gan\*.pth, outputs\diffusion\*.pth, *.py -DestinationPath assignment_submission.zip -Force"
echo Zip created: assignment_submission.zip
