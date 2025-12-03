seed=0
burn_in=6000
thinning=1000
alphas=(3e-7 5e-7 7e-7)

for alpha in "${alphas[@]}"; do
    python main.py --seed "$seed" \
                   --burn_in "$burn_in" \
                   --thinning "$thinning" \
                   --alpha "$alpha"
done
