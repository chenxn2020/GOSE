## In order to reproduce the results of DeMix model on the datasets, you can kindly run the following commands
**Language-specific Fine-tuning:**
```
bash scripts/gose.sh en 3 18265 16
bash scripts/gose.sh zh 4 18265 4
bash scripts/gose.sh ja 4 18265 4
bash scripts/gose.sh es 4 18265 12
bash scripts/gose.sh fr 4 18265 64
bash scripts/gose.sh it 4 18265 4
bash scripts/gose.sh de 4 18265 4
bash scripts/gose.sh pt 4 18265 4
bash scripts/xlm/gose.sh en 4 18265 64 2.5e-5 15 
bash scripts/xlm/gose.sh zh 4 18265 64 2.5e-5 15 
bash scripts/xlm/gose.sh ja 4 18265 4 2.5e-5 15 
bash scripts/xlm/gose.sh es 4 18265 4 2.5e-5 15 
bash scripts/xlm/gose.sh fr 4 18265 4 2.5e-5 15 
bash scripts/xlm/gose.sh it 4 23265 4 2.5e-5 15 
bash scripts/xlm/gose.sh de 4 18265 4 2.5e-5 15 
bash scripts/xlm/gose.sh pt 4 18265 4 2.5e-5 15 
```

**Multilingual fine-tuning:**
```
sh scripts/lilt_multi.sh 4 21995 4 
sh scripts/xlm/multi.sh 4 21995 4 2.5e-5 15
```
