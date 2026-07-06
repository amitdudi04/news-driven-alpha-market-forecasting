@echo off
echo Syncing running project folder to offline backup...
robocopy "G:\News-Driven Alpha Forecasting Chinese Equity Markets using Financial Sentiment" "G:\News-Driven Alpha Offline Backup" /MIR /XD .git .venv venv __pycache__
echo Sync Complete!
pause
