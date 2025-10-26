@echo off
setlocal

:: Activate conda environment
@REM PLEASE EDIT TO USE YOUR OWN ENVIRONMENT PATHS
@REM call C:\Users\xjie\anaconda3\Scripts\activate.bat
@REM call conda activate torch-gpu

echo ==============================================================================
echo Standardized Neural Network Model Training Runner
echo ==============================================================================
echo This will sequentially train NN models for three data sources:
echo   1. Elon Musk Tweets
echo   2. NBC Tesla News Articles
echo   3. NBC Market News Articles
echo Training logs will display in this window. Press Ctrl+C to abort.
echo ==============================================================================
echo.
pause

:: Helper function to run Python script with header/footer
call :RunStage "Elon Musk Tweets" train_tweet_nn.py || goto :eof
call :RunStage "NBC Tesla News" train_tesla_news_nn.py || goto :eof
call :RunStage "NBC Market News" train_market_news_nn.py || goto :eof

echo ==============================================================================
echo All training runs completed. Outputs located in the outputs\ directory.
echo ==============================================================================
echo.
echo Shapley-ready artifacts saved:
echo   - outputs\tweet_shapley_data.npz
echo   - outputs\tesla_news_shapley_data.npz
echo   - outputs\market_news_shapley_data.npz
echo ==============================================================================
goto :end

:RunStage
set "STAGE_NAME=%~1"
set "SCRIPT_NAME=%~2"

echo.
echo ==============================================================================
echo Training Stage: %STAGE_NAME%
echo Script: %SCRIPT_NAME%
echo ==============================================================================
python %SCRIPT_NAME%
if errorlevel 1 (
    echo [ERROR] Stage "%STAGE_NAME%" failed. Aborting.
    pause
    exit /b 1
)
echo [OK] Stage "%STAGE_NAME%" completed.
echo.
pause
exit /b 0

:end
pause
endlocal

