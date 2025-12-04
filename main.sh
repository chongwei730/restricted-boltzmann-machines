burn_in=0
thinning=10
seeds=(1 2 3 4 5 6 7 8 9 10)
for seed in "${seeds[@]}"; do
    python main.py --seed "$seed" \
                   --burn_in "$burn_in" \
                   --thinning "$thinning" 
done