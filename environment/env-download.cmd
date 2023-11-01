@echo off

rem Run as an administrator!!!

call activate wpm-ds-revenue-in-staffing
call conda env list

python -m nltk.downloader punkt
python -m nltk.downloader words
python -m nltk.downloader stopwords

python -m spacy download en
python -m spacy validate
