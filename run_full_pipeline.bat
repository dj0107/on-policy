@echo off
REM run_full_pipeline.bat
REM NALPARI: train -> eval -> visualize (Windows)
REM Usage: run_full_pipeline.bat
REM Prerequisites: conda activate <env>, set ANTHROPIC_API_KEY (optional)

setlocal

REM ============================================================
REM Config  (edit here)
REM ============================================================
set EXP_NAME=nalpari_v1
set NUM_AGENTS=5
set NUM_ENV_STEPS=2000000
set N_ROLLOUT=4
set N_SEEDS=3
set N_EPISODES=3

REM Result paths
set RESULTS_ROOT=onpolicy\scripts\results
set TRAIN_RESULTS=%RESULTS_ROOT%\UAV\uav_tracking\mappo\%EXP_NAME%
REM auto-detect latest run with actor.pt
for /d %%R in ("%TRAIN_RESULTS%\run*") do (
    if exist "%%R\models\actor.pt" set ACTIVEx=%%R
)
if defined ACTIVEx (
    set CHECKPOINT_DIR=%ACTIVEx%\models
    set TRAIN_DONE=%ACTIVEx%\training_done.txt
) else (
    set CHECKPOINT_DIR=%TRAIN_RESULTS%\run1\models
    set TRAIN_DONE=%TRAIN_RESULTS%\run1\training_done.txt
)
set EVAL_OUT=evaluation\%EXP_NAME%
set FIG_OUT=figures\%EXP_NAME%

REM Sweep settings
set SWEEP_UAV=3 5 7 10
set SWEEP_TARGET=1 2 3 4
set SWEEP_NOISE=1 5 20 50
set SWEEP_TAU=0.3 0.5 0.7
set EPISODE_SEEDS=42 100 200

REM ============================================================
REM 0. Checkpoint detection
REM ============================================================
echo.
echo ============================================================
echo   NALPARI Full Pipeline
echo   Experiment: %EXP_NAME%
echo ============================================================
echo.

if exist "%CHECKPOINT_DIR%\actor.pt" goto :has_checkpoint

echo [check] No training checkpoint found. Fresh start required.
goto :fresh_training

:has_checkpoint
if exist "%TRAIN_DONE%" goto :ckpt_done
echo [check] Training checkpoint found but training is NOT complete.
goto :ask_menu

:ckpt_done
echo [check] Training checkpoint found AND training is complete.

:ask_menu
echo.
echo ---
echo What would you like to do?
if exist "%TRAIN_DONE%" (
    echo   [1] Skip training (already done), go to evaluation
) else (
    echo   [1] Resume training from checkpoint
)
echo   [2] Archive existing results and start fresh
echo ---

set CHOICE=
set /p CHOICE=Enter choice (1 or 2): 
if /i "%CHOICE%"=="2" goto :archive_and_start_fresh
if /i "%CHOICE%"=="1" goto :training_entry
echo [warn] Invalid choice; defaulting to option 1.
goto :training_entry

REM ============================================================
REM Archive and start fresh
REM ============================================================
:archive_and_start_fresh
echo.
echo [archive] Archiving existing results...
set TS=%DATE:/=_%
set TS=%TS: =0%
set ARCHIVE_DIR=_archive\%EXP_NAME%_%TS%

if exist "%TRAIN_RESULTS%" (
    if not exist "_archive" mkdir "_archive"
    move "%TRAIN_RESULTS%" "%ARCHIVE_DIR%" 2>nul
    if not errorlevel 1 (
        echo [archive] Training results archived to: %ARCHIVE_DIR%
    ) else (
        echo [warn] Could not move. Cleaning instead...
        rd /s /q "%TRAIN_RESULTS%" 2>nul
    )
)
if exist "%EVAL_OUT%" move "%EVAL_OUT%" "%ARCHIVE_DIR%\evaluation" 2>nul
if exist "%FIG_OUT%" move "%FIG_OUT%" "%ARCHIVE_DIR%\figures" 2>nul
echo [archive] Done. Starting fresh training...
goto :fresh_training

REM ============================================================
REM Training entry points
REM ============================================================
:training_entry
if exist "%TRAIN_DONE%" (
    echo [1/4] Skipping training (already complete).
    goto :evaluation
)
echo [1/4] Resuming training from checkpoint...
set "TRAIN_OPTS=--env_name UAV --scenario_name uav_tracking --algorithm_name mappo --experiment_name %EXP_NAME% --num_agents %NUM_AGENTS% --num_env_steps %NUM_ENV_STEPS% --n_rollout_threads %N_ROLLOUT% --episode_length 100 --use_eval --use_wandb --resume_from %CHECKPOINT_DIR%"
goto :do_training

:fresh_training
echo [1/4] Starting fresh training... (this may take hours)
set "TRAIN_OPTS=--env_name UAV --scenario_name uav_tracking --algorithm_name mappo --experiment_name %EXP_NAME% --num_agents %NUM_AGENTS% --num_env_steps %NUM_ENV_STEPS% --n_rollout_threads %N_ROLLOUT% --episode_length 100 --use_eval --use_wandb"
goto :do_training

:do_training
echo [train] Running: python onpolicy\scripts\train\train_uav.py %TRAIN_OPTS%
set "PYTHONPATH=%cd%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
python onpolicy\scripts\train\train_uav.py %TRAIN_OPTS%
if errorlevel 1 (
    echo [ERROR] Training failed.
    goto :error
)
echo [1/4] Training completed.

REM ============================================================
REM 2. Evaluation
REM ============================================================
:evaluation
echo.
echo [2/4] Evaluating baselines...

echo   [eval] mappo+aai
python tools\evaluate_trained.py --baseline mappo+aai --out_dir "%EVAL_OUT%\mappo_aai" --checkpoint_dir "%CHECKPOINT_DIR%" --sweep_uav %SWEEP_UAV% --sweep_target %SWEEP_TARGET% --sweep_noise %SWEEP_NOISE% --sweep_tau %SWEEP_TAU% --save_episode --episode_seeds %EPISODE_SEEDS% --n_seeds %N_SEEDS% --n_episodes %N_EPISODES%
if errorlevel 1 ( echo [ERROR] mappo+aai eval failed. & goto :error )

echo   [eval] mappo
python tools\evaluate_trained.py --baseline mappo --out_dir "%EVAL_OUT%\mappo_only" --checkpoint_dir "%CHECKPOINT_DIR%" --sweep_uav %SWEEP_UAV% --sweep_target %SWEEP_TARGET% --sweep_noise %SWEEP_NOISE% --sweep_tau %SWEEP_TAU% --save_episode --episode_seeds %EPISODE_SEEDS% --n_seeds %N_SEEDS% --n_episodes %N_EPISODES%
if errorlevel 1 ( echo [ERROR] mappo eval failed. & goto :error )

echo   [eval] naive_greedy
python tools\evaluate_trained.py --baseline naive_greedy --out_dir "%EVAL_OUT%\naive" --sweep_uav %SWEEP_UAV% --sweep_target %SWEEP_TARGET% --sweep_noise %SWEEP_NOISE% --sweep_tau %SWEEP_TAU% --save_episode --episode_seeds %EPISODE_SEEDS% --n_seeds %N_SEEDS% --n_episodes %N_EPISODES%
if errorlevel 1 ( echo [ERROR] naive_greedy eval failed. & goto :error )

echo   [eval] random
python tools\evaluate_trained.py --baseline random --out_dir "%EVAL_OUT%\random" --sweep_uav %SWEEP_UAV% --sweep_target %SWEEP_TARGET% --sweep_noise %SWEEP_NOISE% --sweep_tau %SWEEP_TAU% --save_episode --episode_seeds %EPISODE_SEEDS% --n_seeds %N_SEEDS% --n_episodes %N_EPISODES%
if errorlevel 1 ( echo [ERROR] random eval failed. & goto :error )

if not defined ANTHROPIC_API_KEY (
    echo   [SKIP] llm_aai - ANTHROPIC_API_KEY not set.
    goto :merge
)
echo   [eval] llm_aai
python tools\evaluate_trained.py --baseline llm_aai --out_dir "%EVAL_OUT%\llm_aai" --checkpoint_dir "%CHECKPOINT_DIR%" --aai_callback tools.llm_aai:default_callback --sweep_uav %SWEEP_UAV% --sweep_target %SWEEP_TARGET% --save_episode --episode_seeds %EPISODE_SEEDS% --n_seeds %N_SEEDS% --n_episodes %N_EPISODES%
if errorlevel 1 ( echo [ERROR] llm_aai eval failed. & goto :error )

REM ============================================================
REM 3. Merge results
REM ============================================================
:merge
echo.
echo [3/4] Merging results...
set COMBINED=%EVAL_OUT%\_combined
if not exist "%COMBINED%\sweep_uav" mkdir "%COMBINED%\sweep_uav"
if not exist "%COMBINED%\sweep_target" mkdir "%COMBINED%\sweep_target"
if not exist "%COMBINED%\sweep_noise" mkdir "%COMBINED%\sweep_noise"
if not exist "%COMBINED%\sweep_tau" mkdir "%COMBINED%\sweep_tau"

for %%V in (mappo_aai mappo_only naive random llm_aai) do (
    if exist "%EVAL_OUT%\%%V\sweep_uav\*.npz" copy /Y "%EVAL_OUT%\%%V\sweep_uav\*.npz" "%COMBINED%\sweep_uav\" >nul 2>&1
    if exist "%EVAL_OUT%\%%V\sweep_target\*.npz" copy /Y "%EVAL_OUT%\%%V\sweep_target\*.npz" "%COMBINED%\sweep_target\" >nul 2>&1
    if exist "%EVAL_OUT%\%%V\sweep_noise\*.npz" copy /Y "%EVAL_OUT%\%%V\sweep_noise\*.npz" "%COMBINED%\sweep_noise\" >nul 2>&1
    if exist "%EVAL_OUT%\%%V\sweep_tau\*.npz" copy /Y "%EVAL_OUT%\%%V\sweep_tau\*.npz" "%COMBINED%\sweep_tau\" >nul 2>&1
)

REM ============================================================
REM 4. Visualization
REM ============================================================
echo.
echo [4/4] Generating figures...
if not exist "%FIG_OUT%" mkdir "%FIG_OUT%"

python tools\plot_system_diagram.py --out_dir "%FIG_OUT%"
if errorlevel 1 ( echo [ERROR] plot_system_diagram failed. & goto :error )

python tools\plot_energy_curves.py --results_dir "%COMBINED%" --out_dir "%FIG_OUT%"
if errorlevel 1 ( echo [ERROR] plot_energy_curves failed. & goto :error )

for %%V in (mappo_aai mappo_only naive llm_aai) do (
    if exist "%EVAL_OUT%\%%V\episodes" (
        for %%F in ("%EVAL_OUT%\%%V\episodes\*.npz") do (
            python tools\plot_episode.py --episode_npz "%%F" --out_path "%FIG_OUT%\episode_%%V_%%~nF.png"
        )
    )
)

REM ============================================================
REM Done
REM ============================================================
echo.
echo [done] All artifacts:
echo   Training  : %TRAIN_RESULTS%
echo   Evaluation: %EVAL_OUT%
echo   Figures   : %FIG_OUT%
goto :eof

REM ============================================================
REM Error handler
REM ============================================================
:error
echo.
echo [PIPELINE ABORTED] See error above.
exit /b 1