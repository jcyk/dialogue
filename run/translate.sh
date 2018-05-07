python translate.py --model _out/default/model/epoch15 \
                 --fname ../data/dev

python read.py --fname test_out --o > dev_out
