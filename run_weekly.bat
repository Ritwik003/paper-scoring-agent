@echo off
cd /d "D:\StackMentalist\Reserach Agent part2\paper-scoring-agent-final\project"
set PYTHONPATH=src
call .venv\Scripts\activate.bat
python scripts\run_weekly.py >> logs\weekly.log 2>&1