@echo off
cuda-memcheck 418rad.exe "..\..\vmf\%1"
copy out.bsp D:\Steam\steamapps\sourcemods\BMS\maps\
