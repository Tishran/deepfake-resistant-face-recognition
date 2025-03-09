#!/bin/bash

mkdir ./data
cd ./data

wget https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/train.zip
unzip -q train.zip
rm train.zip

wget https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/test_public.zip
unzip -q test_public.zip
rm test_public.zip

echo "Data is downloaded."