@echo off
cuda-memcheck SilkRAD.exe "..\..\vmf\%1"
copy out.bsp D:\Steam\steamapps\sourcemods\BMS\maps\
