seed=0
burn_in=0
thinning=10
alphas=(3e-7 5e-7 7e-7)

for alpha in "${alphas[@]}"; do
    python main.py --seed "$seed" \
                   --burn_in "$burn_in" \
                   --thinning "$thinning" \
                #    --alpha "$alpha"
done
