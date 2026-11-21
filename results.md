```
teacher trajectories: 300 tasks, 220 kept, 80 dropped by the checker
example: <q> stock of quince plus stock of papaya </q> <call> lookup quince </call> <result> 7 </result> <call> lookup papaya </call> <result> 20 </result> <call> calc 7 + 20 </call> <result> 27 </result> <answer> 27 </answer>

loss on the model's own tokens: 3.39 at the start, 0.000 after 700 steps
loss on tool outputs, which were never a training target: 13.50

benchmark: 60 tasks, 8 trials each, temperature 1.0
tool user, pass@1 of each trial: [0.9333, 0.9167, 0.9333, 0.8667, 0.9333, 0.9167, 0.95, 0.9333]
tool user, step 250    score 0.573 +- 0.040   one run moves by 0.048 between seeds
tool user, step 600    score 0.881 +- 0.040   one run moves by 0.011 between seeds
tool user, step 700    score 0.923 +- 0.031   one run moves by 0.025 between seeds

step 250:  k   pass@k   pass^k
           1   0.573    0.573
           2   0.746    0.400
           4   0.838    0.216
           8   0.867    0.067

step 700:  k   pass@k   pass^k
           1   0.923    0.923
           2   0.941    0.905
           4   0.946    0.879
           8   0.950    0.833

step 250 -> 700: gap +0.350, paired SE 0.034 (unpaired 0.050): new is better
step 600 -> 700: gap +0.042, paired SE 0.030 (unpaired 0.051): cannot tell

tool user: seen 0.933 (n=30), held out 0.912 (n=30), gap +0.021 +- 0.062
memoriser: seen 0.971 (n=30), held out 0.704 (n=30), gap +0.267 +- 0.069

tool user: 0.923 in the training warehouse, 0.950 after the stocks change
memoriser: 0.838 in the training warehouse, 0.358 after the stocks change

tool user with a budget of 2 tool calls instead of 4: 0.633

one episode: <q> stock of quince plus stock of peach </q> <call> lookup quince </call> <result> 28 </result> <call> lookup peach </call> <result> 49 </result> <call> calc 28 + 49 </call> <result> 77 </result> <answer> 77 </answer>
status: answered | correct: True
```
