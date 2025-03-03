Cryptography Datasets

CryptoDL Dataset: https://github.com/deepcrypto/CryptoDL

Contains examples of various encryption algorithms for classification tasks


Kaggle Cryptographic Ciphers Dataset: https://www.kaggle.com/datasets/lstatman/cryptographic-ciphers

Includes multiple cipher types with plaintext-ciphertext pairs


SecLists Wordlists: https://github.com/danielmiessler/SecLists/tree/master/Passwords

Useful for training password cracking components



Steganography Datasets

BOSSbase: http://dde.binghamton.edu/download/

Standard benchmark for steganography detection


Alaska2 Dataset: https://www.kaggle.com/competitions/alaska2-image-steganalysis/data

Large dataset for steganalysis from a Kaggle competition


RAISE Dataset: http://mmlab.science.unitn.it/RAISE/

Raw images for creating steganography examples



Binary Vulnerability Datasets

NIST SARD: https://samate.nist.gov/SARD/

Software Assurance Reference Dataset with known vulnerabilities


Juliet Test Suite: https://samate.nist.gov/SRD/testsuite.php

Thousands of test cases with known vulnerabilities


VulDeePecker Dataset: https://github.com/CGCL-codes/VulDeePecker

Code samples with and without vulnerabilities



Web Vulnerability Datasets

OWASP WebGoat: https://github.com/WebGoat/WebGoat

Deliberately insecure web application for learning


CSIC 2010 HTTP Dataset: https://www.tic.itefi.csic.es/dataset/

Contains normal and anomalous HTTP requests


HTTPCS Dataset: https://github.com/faizann24/Fwaf-Machine-Learning-driven-Web-Application-Firewall

HTTP requests labeled as malicious or benign



General CTF Training Data

CTFd Platform Data: https://github.com/CTFd/CTFd

You can set up your own challenges and collect the data


PicoCTF Archive: https://picoctf.org/

Past CTF challenges that can be used to create training data


CTF Time Writeups: https://ctftime.org/writeups

Contains writeups from past CTF competitions that can be mined for examples



When using these datasets, you'll need to preprocess them to fit the format expected by the training functions in the code. For example, for the cipher classifier, you'd need to extract examples of each cipher type and label them accordingly.
