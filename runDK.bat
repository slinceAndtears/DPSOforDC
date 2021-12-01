@echo off
for /l %%i in (1,1,30) do mpiexec.exe -np 7 python DKmeans.py
pause