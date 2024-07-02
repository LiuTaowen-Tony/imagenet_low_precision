# nearest [doesn't converge]
python imagenet.py \
  --act_man_width 1 --weight_man_width 1 --back_man_width 1 \
  --act_rounding nearest --weight_rounding nearest --back_rounding nearest \
  --back_input_man_width 1 --back_weight_man_width 1 \
  --back_input_rounding nearest --back_weight_rounding nearest \
  --same_input=True --same_weight=True

# stochastic correct [converges]
python imagenet.py \
  --act_man_width 1 --weight_man_width 1 --back_man_width 1 \
  --act_rounding nearest --weight_rounding nearest --back_rounding nearest \
  --back_input_man_width 1 --back_weight_man_width 1 \
  --back_input_rounding nearest --back_weight_rounding nearest \
  --same_input=True --same_weight=True

# nearest 2 [converges]
python imagenet.py \
  --act_man_width 2 --weight_man_width 2 --back_man_width 2 \
  --act_rounding nearest --weight_rounding nearest --back_rounding nearest \
  --back_input_man_width 2 --back_weight_man_width 2 \
  --back_input_rounding nearest --back_weight_rounding nearest \
  --same_input=True --same_weight=True

# stochastic incorrect
python imagenet.py \
  --act_man_width 2 --weight_man_width 2 --back_man_width 2 \
  --act_rounding stochastic --weight_rounding stochastic --back_rounding stochastic \
  --back_input_man_width 2 --back_weight_man_width 2 \
  --back_input_rounding stochastic --back_weight_rounding stochastic \
  --same_input=True --same_weight=True

# stochastic correct
python imagenet.py \
  --act_man_width 2 --weight_man_width 2 --back_man_width 2 \
  --act_rounding stochastic --weight_rounding stochastic --back_rounding stochastic \
  --back_input_man_width 2 --back_weight_man_width 2 \
  --back_input_rounding stochastic --back_weight_rounding stochastic \
  --same_input=False --same_weight=False

# nearest navin 2 3
python imagenet.py \
  --act_man_width 3 --weight_man_width 3 --back_man_width 2 \
  --act_rounding nearest --weight_rounding nearest --back_rounding nearest \
  --back_input_man_width 2 --back_weight_man_width 2 \
  --back_input_rounding nearest --back_weight_rounding nearest \
  --same_input=False --same_weight=False

# nearest 3 bits
python imagenet.py \
  --act_man_width 3 --weight_man_width 3 --back_man_width 3 \
  --act_rounding nearest --weight_rounding nearest --back_rounding nearest \
  --back_input_man_width 3 --back_weight_man_width 3 \
  --back_input_rounding nearest --back_weight_rounding nearest \
  --same_input=True --same_weight=True

# stochastic 3 bits
python imagenet.py \
  --act_man_width 3 --weight_man_width 3 --back_man_width 3 \
  --act_rounding stochastic --weight_rounding stochastic --back_rounding stochastic \
  --back_input_man_width 3 --back_weight_man_width 3 \
  --back_input_rounding stochastic --back_weight_rounding stochastic \
  --same_input=False --same_weight=False