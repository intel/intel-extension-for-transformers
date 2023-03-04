# from http://nlp.cs.washington.edu/triviaqa/  and https://github.com/mandarjoshi90/triviaqa
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz

tar -xvzf triviaqa-rc.tar.gz

# the blow codes from the original paper code: https://github.com/allenai/longformer
python -m utils.convert_to_squad_format  \
  --triviaqa_file ./qa/wikipedia-train.json  \
  --wikipedia_dir ./evidence/wikipedia/   \
  --web_dir ./evidence/web/  \
  --max_num_tokens 4096  \
  --squad_file squad-wikipedia-train-4096.json

python utils.convert_to_squad_format  \
  --triviaqa_file ./qa/wikipedia-dev.json  \
  --wikipedia_dir ./evidence/wikipedia/   \
  --web_dir ./evidence/web/  \
  --max_num_tokens 4096  \
  --squad_file squad-wikipedia-dev-4096.json
