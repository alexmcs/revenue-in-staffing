@echo off

call activate wpm-ds-revenue-in-staffing

call conda env update -n=wpm-ds-revenue-in-staffing -f env.yml
cmd /k
