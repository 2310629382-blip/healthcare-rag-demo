@echo off
chcp 65001 >nul
cd /d %~dp0
echo 正在启动社区康养知识库问答 Demo ...
python -m streamlit run app.py
pause